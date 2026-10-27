"""Tests for the ImageToolbox file utilities (tools/file_tools.py): cipher, ZIP archives, audio cover
extraction, code preview, document scanner (see docs/dev/port_reports/file_tools.md)."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import lazyroom.tools.file_tools  # noqa: F401  (registers the tools)
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.core.io import save_image
from lazyroom.core.params import validate_params
from lazyroom.tools import _archive as A
from lazyroom.tools import _audio_cover as AC
from lazyroom.tools import _cipher as C
from lazyroom.tools import _code_preview as CP
from lazyroom.tools import _docscan as D
from lazyroom.tools.registry import TOOLS

pytest.importorskip("cryptography")
try:
    import pygments  # noqa: F401  (optional: code preview highlighting)
    HAS_PYGMENTS = True
except ImportError:  # pragma: no cover
    HAS_PYGMENTS = False


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


def run(api, _tool, **params):
    t = TOOLS[_tool]
    return t.fn(api, **validate_params(t.params, params, op_id=_tool))


def err(api, code, _tool, **params) -> LazyRoomError:
    with pytest.raises(LazyRoomError) as e:
        run(api, _tool, **params)
    assert e.value.code == code, e.value
    return e.value


# ===========================================================================
# Cipher
# ===========================================================================
PASSWORD = "ImageToolbox"
BC_PLAINTEXT = b"ImageToolbox Add unsupported ciphers to CipherType.BROKEN"

# Produced by the real BouncyCastle provider (bcprov 1.77, JDK 25) running ImageToolbox's
# AndroidCryptographyManager code paths (versioned: fixed salt/nonce, 1000 iterations; legacy: createKey +
# Cipher.init fallbacks).  Password "ImageToolbox", plaintext BC_PLAINTEXT.
BC_VERSIONED = {
    'AES/GCM/NoPadding': "49544258435259500101000003e80100100c001101080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaab4145532f47434d2f4e6f50616464696e679030552c15131f0aa534f05699b64680b8aef355c73400d35cf0c4b6acb2918dd0f899bf4d01ffdca600fe00c6241d0428fdff9725f769ca15164a1687befc5941105eb090f9f8144f",
    'AES': "49544258435259500101000003e801001010000301080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf414553259dc2c16c8e8cf2c0de82e81c05ee2ba89775eebb5fd9466f06532e17bd0de7fdcf94fb5e0c5538eb1666c9d3a2bf48eb7f660da5e5ac2511fc9f5c55e30c76",
    'AES/CTR/NOPADDING': "49544258435259500101000003e801001010001101080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf4145532f4354522f4e4f50414444494e476cbd4f170e988233c23e55ebb950f67b77f27c84052641bdb84014cea228ebfd8ddf2ac98cd0da9cb689cfb2b2b686152c2ef40a13a1cb3409",
    'AES/ECB/PKCS5PADDING': "49544258435259500101000003e801001010001401080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf4145532f4543422f504b43533550414444494e47259dc2c16c8e8cf2c0de82e81c05ee2ba89775eebb5fd9466f06532e17bd0de7fdcf94fb5e0c5538eb1666c9d3a2bf48eb7f660da5e5ac2511fc9f5c55e30c76",
    'AES/GCM/NOPADDING': "49544258435259500101000003e80100100c001101080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaab4145532f47434d2f4e4f50414444494e479030552c15131f0aa534f05699b64680b8aef355c73400d35cf0c4b6acb2918dd0f899bf4d01ffdca600fe00c6241d0428fdff9725f769ca15c154e2c6c03e29bf73edc49365995da4",
    'AES_256/ECB/PKCS5PADDING': "49544258435259500101000003e801001010001801080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf4145535f3235362f4543422f504b43533550414444494e47259dc2c16c8e8cf2c0de82e81c05ee2ba89775eebb5fd9466f06532e17bd0de7fdcf94fb5e0c5538eb1666c9d3a2bf48eb7f660da5e5ac2511fc9f5c55e30c76",
    'ARC4': "49544258435259500101000003e801001010000401080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf41524334541f3dbf3956ab4d0bb6801a23a3185f2b414e9d96b16a1f2a8fae51b7b048f79709b70a58235afff67f2cea93d7cc31380754d8b95ecf9530",
    'BLOWFISH': "49544258435259500101000003e801001010000801080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf424c4f57464953480c70ca8e504d7f8140771726fc1bd6a8b03e407f2a26cc310afb779512ded9ea28325a2af2c2b6d1df494586ea9c3c5b203279baf4c612b397461f785534c5c8",
    'CAMELLIA': "49544258435259500101000003e801001010000801080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf43414d454c4c494198009f716a61bbe1d0a549eb7d44d0b0d56743b517deb5eb5890b1bd33c541a439fd7e6ed259d5ae53dca980a8671ba5807a614ebf09225fd1e02b71f7d7d619",
    'DESEDE': "49544258435259500101000003e800801010000601080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf444553454445b0513594d66c097f4d1b25fa830a68ca81504bba68236a8469a74629af4b8293dec708ca6f09a85abd50140070fc8dcb531ddada70b3d05d469693dc9223c5a9",
    'DESEDE/CBC/PKCS5PADDING': "49544258435259500101000003e800801008001701080f161d242b323940474e555c636aa0a1a2a3a4a5a6a74445534544452f4342432f504b43533550414444494e47ed4a5721edd6fe6f8a7d55eb2fdd2d6fd871b781d1314fdecff7de26620cb47d8caea4b5f67c0d767dbcb1564977a80194e83fe824574493d940a370aab6b316",
    'DESede/CBC/NoPadding': "49544258435259500101000003e800801008001201080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7312e322e3834302e3131333534392e332e37ed4a5721edd6fe6f8a7d55eb2fdd2d6fd871b781d1314fdecff7de26620cb47d8caea4b5f67c0d767dbcb1564977a80194e83fe824574493d940a370aab6b316",
    'GCM': "49544258435259500101000003e80100100c000301080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaab47434d9030552c15131f0aa534f05699b64680b8aef355c73400d35cf0c4b6acb2918dd0f899bf4d01ffdca600fe00c6241d0428fdff9725f769ca15c06a926c73ddf1b531df8f1c6796680a",
    'IDEA': "49544258435259500101000003e801001010000401080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf49444541c5d6e9e194ca45b6756a82ce629dcf4fc1774a1cf334c781e510f216464fa626eed52bc78871d10120375c7098628e38e50314970f125a4a49904c6ba4f82c90",
    'PBEWITHMD5AND128BITAES-CBC-OPENSSL': "49544258435259500101000003e801001008002201080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7504245574954484d4435414e443132384249544145532d4342432d4f50454e53534cba93f1838b64d382a90e572179d6df7760bde785abfe5ad664cefbd6ace0a2f61f371b894b37684eebcb2d5c03ead081c16e5d0fe55853754be581aecf55ea4a",
    'PBEWITHMD5AND192BITAES-CBC-OPENSSL': "49544258435259500101000003e801001008002201080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7504245574954484d4435414e443139324249544145532d4342432d4f50454e53534c9fd6e3423f41a3889271dde14c98ad7f8c43ee54da0e83fa975432462bbc0022fab5255ee87cea4aad5bc931d6d3ed6ca9162bce3093af234d26bb22045b4836",
    'PBEWITHMD5AND256BITAES-CBC-OPENSSL': "49544258435259500101000003e801001008002201080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7504245574954484d4435414e443235364249544145532d4342432d4f50454e53534c899522c488392426b1093cc8a56aa3973157b1964d4c5e5c01066e56b5d1dafaeac543720fd147d6efb5610af09a854d7fba30d36ff4fb54c2e3ee89a4574b1e",
    'PBEWITHMD5ANDDES': "49544258435259500101000003e801001010001001080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf504245574954484d4435414e4444455384737f4175d7cc2c7bc4a916b25aa4c57c3f214ed41137c8488be3a8f38cfcd42de0728bbd42b9990f21494265154063f05720388a2e0061329257cec5426935",
    'PBEWITHSHA1ANDDES': "49544258435259500101000003e801001010001101080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf5042455749544853484131414e44444553cfaf393e9b562e2180d9fe4fc116468ca360ccb9e0a84b8367285c0e2052cde8bd669308e17d4687b62303384ccc073d1760dd1a5317eed130b6dd86e580964c",
    'PBEWITHSHA256AND128BITAES-CBC-BC': "49544258435259500101000003e801001008002001080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841323536414e443132384249544145532d4342432d42430a8b715c31a2135ced07a0a322a2aec5e45637f480bfee755f103164fae10c1432f0dc4540aaed0f35dec9452394990bb7bc13e5ceca30a8f7577fedf0ccb9fe",
    'PBEWITHSHA256AND192BITAES-CBC-BC': "49544258435259500101000003e801001008002001080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841323536414e443139324249544145532d4342432d4243660eb13709e42f593d597ee621c3d2402995bd7215d6af4bfc989362cf5c8db2d30528d5208633e5053dd24b7a2cdd7cd0b3b9e401ec83f0388f7f7f2da21aa5",
    'PBEWITHSHA256AND256BITAES-CBC-BC': "49544258435259500101000003e801001008002001080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841323536414e443235364249544145532d4342432d42430b2b886c6d70ee50474d9ba7518919083b8357314d92b6b37b745aeaf02c2247f319cbddbb48fa6b3f0fb7975436f1fc2259786fcd3e5d0cc559d19d013bec75",
    'PBEWITHSHAAND128BITAES-CBC-BC': "49544258435259500101000003e801001008001d01080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841414e443132384249544145532d4342432d424380c55986f09c38782a9ea4d85b84a5b0a8a5e3a54a2111fac6fe23847ce5ff63012929d51aa16c44cec6c76db3d2447fa1e3fb3f7dfa1dfe53545a5887e7832f",
    'PBEWITHSHAAND128BITRC2-CBC': "49544258435259500101000003e801001008001a01080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841414e443132384249545243322d4342434375c80f997b6995e5e9fb450d3f8cbfb20cc2c35d8f904bd639cb1d5c00cfbcc3390764f8f7320c7800d90d3f9de1bf196dd2121760dc8e1f021d0b40573203",
    'PBEWITHSHAAND128BITRC4': "49544258435259500101000003e801001010001601080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf50424557495448534841414e443132384249545243342d9ad0a1dac632e250f82a6e6feb1f0d59643e9a7a79df1d474b81491c44ec8e7eb2529439580b700b25b756088a0fc752b5d00c10778c9069",
    'PBEWITHSHAAND192BITAES-CBC-BC': "49544258435259500101000003e801001008001d01080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841414e443139324249544145532d4342432d4243e90a0b345f6e8243699ad3d1c0311985b5172909747d4ff9ed515af46c34edb127f55e6b9fbb684ee5ba755f5a003c7af38041f569d4d7f23beabb534e612d51",
    'PBEWITHSHAAND2-KEYTRIPLEDES-CBC': "49544258435259500101000003e801001008001f01080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841414e44322d4b4559545249504c454445532d4342439128e734b14283183da4d5267543cbd86aedbf3b15c2cd97e22b5a8965378b6d8f3af36694c9f9a4a02e6196314f59116080b3df89ec5dc1f47e6340750941f0",
    'PBEWITHSHAAND256BITAES-CBC-BC': "49544258435259500101000003e801001008001d01080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841414e443235364249544145532d4342432d4243a59749736dba466a103d65cd0f7697f10c9d97ca48c1b7150fbe91b52bdc7dd52921fd78ee78e6b2fa796cf66d5a153c5348d88dfa963f5572ab5cda12452356",
    'PBEWITHSHAAND3-KEYTRIPLEDES-CBC': "49544258435259500101000003e801001008001f01080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841414e44332d4b4559545249504c454445532d434243d2f4f31333bbc0b36cb21bc578c15133ac52d118cd75450b4b1768cfc6d59bbfb7f4c6ed0316d9bf53066f16f169c548501f7cefcef165a6cae0d88b0594d09f",
    'PBEWITHSHAAND40BITRC4': "49544258435259500101000003e801001010001501080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf50424557495448534841414e44343042495452433442c8855e24e06dad234e9dde9033d19f418281ac26ffcc0b9325baf85be118770857a3d2a323f3b23a41e5dc675c1ec0d55569fee0d6472a8f",
    'PBEWITHSHAANDIDEA-CBC': "49544258435259500101000003e801001008001501080f161d242b323940474e555c636aa0a1a2a3a4a5a6a750424557495448534841414e44494445412d4342432b602c65f423bcee81bc33ff608d0fc2805942e330439eb1d47237521ee8d73e97b32ac545ee3745428141278ff18dd3ff3c5672a51c6824890c3874b9191cb5",
    'RIJNDAEL': "49544258435259500101000003e801001010000801080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf52494a4e4441454c259dc2c16c8e8cf2c0de82e81c05ee2ba89775eebb5fd9466f06532e17bd0de7fdcf94fb5e0c5538eb1666c9d3a2bf48eb7f660da5e5ac2511fc9f5c55e30c76",
    'SEED': "49544258435259500101000003e800801010000401080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf53454544eff8bc0443c691a53260250e994cccdae485ae1c024a7ba73949d0b00f6b43fdcd5d306df802b692ed76f0cb1d6f808bdfcd6b79e164d10911dcaae3a0915b44",
    'SM4': "49544258435259500101000003e800801010000301080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaabacadaeaf534d34a100603093d161676af493720c2879603491381a103bec1a35c87d338a34220c5d3c3021cbb6f7ef1bc532cf8a2b9bba5facd999d6ca35b10d1dabcac5f630d0",
}
BC_LEGACY = {
    'AES/GCM/NoPadding': "c30672df1ba3ff107e87d442914ea241a7b0183b94354e1db9bf34aa3af75037febb9771fdff0097c7ae24264f7680a1b5e46cc9cb4d219d140940ea0818375235767413f920e8f538",
    'AES': "ac58d2eab161b84f8a7a4e82030da0eab40f24b5e9995d6390f06f5ee11e811384b1ee343fe709b4874889358b10625c65e7b19568bb199264338880bd963a6c",
    'AES/CTR/NOPADDING': "c174441bbed27440e4ead707d9a1f7510d15ba145ade7f0c135c0c58da12db4d2b8e760c4c7bd7500ffd561c2ac98a15017f09fabfcf75b5ba",
    'AES/ECB/PKCS5PADDING': "ac58d2eab161b84f8a7a4e82030da0eab40f24b5e9995d6390f06f5ee11e811384b1ee343fe709b4874889358b10625c65e7b19568bb199264338880bd963a6c",
    'AES/GCM/NOPADDING': "c30672df1ba3ff107e87d442914ea241a7b0183b94354e1db9bf34aa3af75037febb9771fdff0097c7ae24264f7680a1b5e46cc9cb4d219d140940ea0818375235767413f920e8f538",
    'AES_256/ECB/PKCS5PADDING': "ac58d2eab161b84f8a7a4e82030da0eab40f24b5e9995d6390f06f5ee11e811384b1ee343fe709b4874889358b10625c65e7b19568bb199264338880bd963a6c",
    'ARC4': "583f29d27d8ad0163b6ef20cd1d3f75b839da8ea00297607c0a1cee4f9407eb65d17d6a6c12c9ac4a9f28976532714d502c5d65956c67e58da",
    'BLOWFISH': "44cbfe03d3cf13e8825202753ba18d8d9bf66f3a4c8350974e09b9f8d65c860562f0ab433ce69631adbef1a5b0aae3aafbfa5d2e5233f16db035768de9b68953",
    'CAMELLIA': "bd174b628db0577714071bf6be64795f22f1956f7331423391be8bb55f7b78255408a95f65d1f8042769e02188f74bcd9cd9e44addb84db45943a416c6a87c87",
    'DESEDE': "7b69e4d4689cd3d04c1f3fd488af3f00842304b9fbb3ec3c693bbcc3f95440e5ed417483aa0b9a769be18ff110b0f3f18b4c10c040be63d964f1a024cb59d348",
    'DESEDE/CBC/PKCS5PADDING': "dc55df77e9c009b8c079418c90b8c63ef9de3b2eb7795a77a54c69c31d8fa500a74f5d3ca61908da5e7c149861e34264abb03515c1f695b78719d6f69442791b",
    'DESede/CBC/NoPadding': "dc55df77e9c009b8c079418c90b8c63ef9de3b2eb7795a77a54c69c31d8fa500a74f5d3ca61908da5e7c149861e34264abb03515c1f695b78719d6f69442791b",
    'GCM': "c30672df1ba3ff107e87d442914ea241a7b0183b94354e1db9bf34aa3af75037febb9771fdff0097c7ae24264f7680a1b5e46cc9cb4d219d140940ea0818375235767413f920e8f538",
    'IDEA': "5b982bf342e3b14718e16e7e63715eddf0765df465248148a094d285a0f7488f653e5602cbed4e7ec61a155c90970f66d5277b0a0a06e4ef43fea550cca9732f",
    'PBEWITHMD5AND128BITAES-CBC-OPENSSL': "7a60ad3b78b8b61372c353314f3cfb22695555c1a8a4eca454ab696e52c08e2c4055351370df1545a8c8fd3aa99e2b0a664d273f0f08705df5605af0f9f25ed2",
    'PBEWITHMD5AND192BITAES-CBC-OPENSSL': "026ec5ab4b64674423490b8ebac714a58cc0033b244b9e9b81fb96d6c53a006c8bca40c56222fadb12463cd29ab590d395469c85f3a7e55015c6417c7507c481",
    'PBEWITHMD5AND256BITAES-CBC-OPENSSL': "da287f52f31f62a895784742d1218b558842a069f5d36e32316b579cfa5994a88255f8796e09a9f7b1941180e7f39e9f842e71376fba389acbdafad70ca6f735",
    'PBEWITHMD5ANDDES': "852c97a44a742bf0c390e88b94d2f64e03a55f4840d500c6702a0b836bfdef8a83a0ad78c3faabefbc901b25f5674c081ae6956af921f7ed608b6d4cda3ac557",
    'PBEWITHSHA1ANDDES': "9ca0a27b65c4088b76205a430b88a45485922cb7285a68ca80daf45ea6e13033d2d94448d0c0c19d810cea32b054a1bce2961a9f59a7cf841bcf677fc6b27ae3",
    'PBEWITHSHA256AND128BITAES-CBC-BC': "c9292768b9aa383d25768d553b2a49f6ed80a8f3a030f0d758742cf1c0f9c75d38bc4988bf80e61405e79a96595cf7e3362dc41cb18564e66b810d80608bf072",
    'PBEWITHSHA256AND192BITAES-CBC-BC': "c343722f8507fef76fc2d7504140690667f564c9d0b6454c060b86f5ee015fa8bc2c703d6bbc336fb723dc13ce10fd3f0d65f33f681ae3ed680994a0fb6e930b",
    'PBEWITHSHA256AND256BITAES-CBC-BC': "ae4d71864967d14c5e8ced8558f63722104acab5badcc80f9c61dfabddf002d6ebda3e55e9ddc2284185857dcf463452a840d959c4697e04191d6d9c16bed844",
    'PBEWITHSHAAND128BITAES-CBC-BC': "99f13f9b12385895a7cb53e4fca65670548e271c610cf76bac448423d8d0164984b7248411a85bc9987145eb3b5a37b7524411622c9aed5477c9c53259a48727",
    'PBEWITHSHAAND128BITRC2-CBC': "7755abebd36c91393c0497511e320d9935fc29d3fc8ccb4412e27faf319cb73ec0f91e545ef1be875ce8a2f7b3f8c1169fa2759968dd744eb0310b32df205209",
    'PBEWITHSHAAND128BITRC4': "bcbafba8aebb32f1d46ac8a90550a75a35d8107d3652442c7fd3b2311eff14bb68f9699a4e4fd0cb506f6df03337901b5757e1e71bd662ee45",
    'PBEWITHSHAAND192BITAES-CBC-BC': "552efc8898ada2c52b1b179e4311cc96ac46535ae8cd6ac1c1912a38aa8d982b209e9ae3af1544b3bd4a4544cb2d646830c992202de44397e7346d48e35643e5",
    'PBEWITHSHAAND2-KEYTRIPLEDES-CBC': "3e6a8dd8b9a51cb07fbe1ef8f413b58010785da76a58297ed019bc547de8b05eae01bf4ecc61df6db56045154ebcb9e409a340eb777eba60a29c1baf05141ea3",
    'PBEWITHSHAAND256BITAES-CBC-BC': "f5b66a2e6d200193562faff399ef808a5a04a460cc1f9779b623761948d8180b0425e1e56651663751a85c29b2eeac67e94b4a07e09667ff3c216c837da12198",
    'PBEWITHSHAAND3-KEYTRIPLEDES-CBC': "a7218537daf315c583d1755641f6f4a7698f704c501a45244ce51bd90a4e872095655ab49b1384464a812f745255b34ebec02af3131d9da8b4d105d1a802f3af",
    'PBEWITHSHAAND40BITRC4': "46751aa58d4981df34cc05bd8831f3df104c58f8d4271b84d5610e3a05d8f8e91816aeabfefdca36db563a182bf5e8188e998b0df93f81c78f",
    'PBEWITHSHAANDIDEA-CBC': "ba5ad7693ff90e2ab58faa129ceb964e6e03c38834c56e780c1bf2c2d47c18373233c68259c6ed37a4fa64e51d9451f1cdfe0c7c33dfc95287d11b374772a60d",
    'RIJNDAEL': "ac58d2eab161b84f8a7a4e82030da0eab40f24b5e9995d6390f06f5ee11e811384b1ee343fe709b4874889358b10625c65e7b19568bb199264338880bd963a6c",
    'SEED': "0a018184308a4885b06c35473bf1aadcf8fc021a02d57727326e416ab7acffcd8128509260dd0ab35555591587ab139fcbd6c76efc9afdcb1262824f14207e4e",
    'SM4': "d11df9da03291c1da1d6e540c3b1bba9036dcdba5881f6de3e660a08c5a928faec1bd257b49007e16cb8d891a81f4dc3de9f71075f86757ff0159cefe4620630",
}
BC_DEFAULT_210000 = "49544258435259500101000334500100100c001101080f161d242b323940474e555c636aa0a1a2a3a4a5a6a7a8a9aaab4145532f47434d2f4e6f50616464696e67b15c4df3b454e3d78836a7cfd4e7e35ee611acedbf11947e0ecfd762e1b9c4ff8056c0b62af2c7a2067889b08f29a6a2c7a951f59db09510169bc74f0595f4c728a07df358231d135d"


@pytest.mark.parametrize("name", C.SUPPORTED_NAMES)
def test_bouncycastle_versioned_fixture_decrypts(name):
    out, info = C.decrypt(bytes.fromhex(BC_VERSIONED[name]), PASSWORD)
    assert out == BC_PLAINTEXT
    assert info["format"] == "versioned" and info["algorithm"] == name


@pytest.mark.parametrize("name", C.SUPPORTED_NAMES)
def test_bouncycastle_legacy_fixture_decrypts(name):
    out, info = C.decrypt(bytes.fromhex(BC_LEGACY[name]), PASSWORD, name)
    assert out == BC_PLAINTEXT and info["format"] == "legacy"


def test_bouncycastle_default_fixture_with_full_iterations():
    data = bytes.fromhex(BC_DEFAULT_210000)
    header, size = C.decode_header(data)
    assert header.iterations == 210_000 and header.cipher == "AES/GCM/NoPadding" and header.key_size == 256
    assert C.decrypt(data, PASSWORD)[0] == BC_PLAINTEXT


@pytest.mark.parametrize("name", C.SUPPORTED_NAMES)
def test_round_trip_every_supported_algorithm(name):
    data = "Grüße – any length works, even 0x".encode() * 3
    blob = C.encrypt(data, "pässwörd", name, iterations=1000)
    header, size = C.decode_header(blob)
    d = C.lookup(name)
    assert blob[:8] == b"ITBXCRYP" and blob[8] == 1 and blob[9] == 1
    assert header.cipher == d.cipher and len(header.salt) == 16
    assert len(header.nonce) == (12 if C.is_gcm(name) else C.iv_size(name))
    assert header.key_size == C.key_size_bits(name)
    out, info = C.decrypt(blob, "pässwörd")
    assert out == data and info["algorithm"] == name


def test_round_trip_empty_input():
    blob = C.encrypt(b"", PASSWORD, iterations=10)
    assert C.decrypt(blob, PASSWORD)[0] == b""


def test_imagetoolbox_test_vectors():
    # repeatedEncryptionUsesUniqueSaltAndNonce
    a, b = (C.encrypt(BC_PLAINTEXT, PASSWORD, iterations=10) for _ in range(2))
    assert a != b and a[20:36] != b[20:36] and a[36:48] != b[36:48]
    # versionedFileCanBeDecryptedWithoutMatchingSelectedCipher
    assert C.decrypt(a, PASSWORD, "DESEDE")[0] == BC_PLAINTEXT
    # incorrectPasswordIsRejected
    with pytest.raises(LazyRoomError) as e:
        C.decrypt(a, "incorrect")
    assert e.value.code == ErrorCode.DECODE_FAILED and e.value.message == C.DECRYPT_FAILED
    # corruptedHeaderAndCiphertextAreRejected
    bad_header = bytearray(a)
    bad_header[8] = 2
    bad_body = bytearray(a)
    bad_body[-1] = (bad_body[-1] + 1) % 256
    for corrupted in (bad_header, bad_body):
        with pytest.raises(LazyRoomError) as e:
            C.decrypt(bytes(corrupted), PASSWORD)
        assert e.value.message == C.DECRYPT_FAILED
    # legacyFileCanStillBeDecrypted: SHA-256 key, IV = first block of the key, no AAD
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = hashlib.sha256(PASSWORD.encode()).digest()
    legacy = AESGCM(key).encrypt(key[:16], BC_PLAINTEXT, None)
    assert C.decrypt(legacy, PASSWORD, "AES/GCM/NoPadding")[0] == BC_PLAINTEXT
    # encryptedFileCanBeDetected
    assert C.has_magic(a) and not C.has_magic(BC_PLAINTEXT)


def test_header_validation_rules():
    blob = C.encrypt(b"x", PASSWORD, iterations=10)
    too_many = bytearray(blob)
    too_many[10:14] = struct.pack(">i", 210_001)
    for bad in (blob[:19], bytes(too_many), blob[:C.decode_header(blob)[1]]):
        with pytest.raises(LazyRoomError):
            C.decrypt(bad, PASSWORD)


def test_cipher_tools_round_trip_and_naming(api, tmp_path):
    src = tmp_path / "photo.bin"
    src.write_bytes(bytes(range(256)) * 9)
    r = run(api, "encrypt_file", input=str(src), password="s3cret!")
    assert Path(r["output"]).name == "enc-photo.bin" and r["iterations"] == 210_000 and r["authenticated"]
    d = run(api, "decrypt_file", input=r["output"], password="s3cret!")
    assert Path(d["output"]).name == "dec-enc-photo.bin"  # ImageToolbox: "dec-" + file name
    assert Path(d["output"]).read_bytes() == src.read_bytes()
    err(api, ErrorCode.INVALID_PARAMS, "decrypt_file", input=r["output"], password="s3cret!", output=r["output"],
        overwrite=True)
    e = err(api, ErrorCode.INVALID_PARAMS, "encrypt_file", input=str(src), password="bad\udcff",
            output=str(tmp_path / "x"))
    assert "udcff" not in json.dumps(e.to_dict()).lower()
    # encrypting an encrypted file warns (additional layer)
    again = run(api, "encrypt_file", input=r["output"], password="x", algorithm="AES", output=str(tmp_path / "e2"))
    assert "already encrypted" in again["warning"] and "note" in again
    err(api, ErrorCode.OUTPUT_EXISTS, "encrypt_file", input=str(src), password="s3cret!")
    err(api, ErrorCode.INVALID_PARAMS, "encrypt_file", input=str(src), password="")
    err(api, ErrorCode.INVALID_PARAMS, "encrypt_file", input=str(src), password="x", algorithm="TWOFISH")


def test_wrong_password_is_a_clean_error_and_writes_nothing(api, tmp_path):
    src = tmp_path / "a.txt"
    src.write_bytes(b"hello")
    enc = run(api, "encrypt_file", input=str(src), password="right")["output"]
    e = err(api, ErrorCode.DECODE_FAILED, "decrypt_file", input=enc, password="wrong-password")
    assert "invalid password" in e.message
    assert "wrong-password" not in json.dumps(e.to_dict())
    assert not (tmp_path / "dec-enc-a.txt").exists()


def test_password_never_reaches_the_operation_log(api, tmp_path):
    src = tmp_path / "doc.txt"
    src.write_bytes(b"confidential")
    secret, wrong = "Tr0ub4dor&3-UNIQUE-PW", "Wr0ng-UNIQUE-PW"
    enc = api.tool_run("encrypt_file", {"input": str(src), "password": secret})["output"]
    api.tool_run(name="decrypt_file", params={"input": enc, "password": secret, "output": str(tmp_path / "o1")})
    for params in ({"input": enc, "password": wrong, "output": str(tmp_path / "o2")},
                   {"input": enc, "password": secret, "algorithm": "bogus"},
                   {"input": enc, "password": secret, "unknown": 1}):
        with pytest.raises(LazyRoomError) as e:
            api.tool_run("decrypt_file", params)
        assert secret not in json.dumps(e.value.to_dict()) and wrong not in json.dumps(e.value.to_dict())
    gen = api.tool_run("generate_password", {"length": 24})["password"]
    log = Path(api.ws.log.path).read_text(encoding="utf-8")
    assert log.count('"command": "tools.run"') >= 6
    for s in (secret, wrong, gen):
        assert s not in log


def test_cipher_algorithms_tool(api):
    r = run(api, "cipher_algorithms")
    assert r["backend_installed"] and r["total"] == len(C.CIPHERS) and r["supported"] == len(C.SUPPORTED_NAMES)
    assert r["algorithms"][0]["name"] == "AES/GCM/NoPadding" and r["default"] == "AES/GCM/NoPadding"
    for row in r["algorithms"]:
        if row["supported"]:
            assert row["available"], row
        else:
            assert row["reason"], row
    assert all(x["supported"] for x in run(api, "cipher_algorithms", include_unsupported=False)["algorithms"])


def test_generate_password(api):
    r = run(api, "generate_password")
    assert r["length"] == 18 and len(r["password"]) == 18 and r["password"].isalnum() and r["password"].isascii()
    assert run(api, "generate_password", length=64)["password"] != run(api, "generate_password", length=64)["password"]


def test_missing_cryptography_is_reported(api, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cryptography.hazmat.primitives.ciphers", None)
    src = tmp_path / "a.txt"
    src.write_bytes(b"x")
    e = err(api, ErrorCode.BACKEND_UNAVAILABLE, "encrypt_file", input=str(src), password="pw")
    assert "pip install cryptography" in e.hint
    assert run(api, "cipher_algorithms")["backend_installed"] is False


# ===========================================================================
# Archives
# ===========================================================================

def _files(tmp_path: Path) -> list[str]:
    (tmp_path / "in1").mkdir()
    (tmp_path / "in2").mkdir()
    a = tmp_path / "in1" / "Photo.png"
    b = tmp_path / "in2" / "photo.png"
    c = tmp_path / "notes.txt"
    a.write_bytes(b"A" * 5000)
    b.write_bytes(b"B" * 3000)
    c.write_text("hello " * 100)
    return [str(a), str(b), str(c)]


@pytest.mark.parametrize("compression", ["store", "deflate", "bzip2", "lzma"])
def test_archive_create_list_extract(api, tmp_path, compression):
    files = _files(tmp_path)
    out = tmp_path / "pack.zip"
    r = run(api, "archive_create", inputs=files, output=str(out), compression=compression, level="maximum")
    assert r["entries"] == ["Photo.png", "photo (2).png", "notes.txt"]
    with zipfile.ZipFile(out) as zf:
        assert [i.filename for i in zf.infolist()] == r["entries"]
        assert all((i.external_attr >> 16) & 0o777 == 0o644 for i in zf.infolist())
        assert zf.read("photo (2).png") == b"B" * 3000
    lst = run(api, "archive_list", input=str(out))
    assert lst["count"] == 3 and lst["encryption"] == "none" and lst["total_size"] == 5000 + 3000 + 600
    x = run(api, "archive_extract", input=str(out), output_dir=str(tmp_path / "dest"))
    root = Path(x["output_dir"])
    assert root.name == "pack" and x["extracted"] == 3
    assert (root / "photo (2).png").read_bytes() == b"B" * 3000
    # extracting again never overwrites: new subfolder "pack (2)"
    assert Path(run(api, "archive_extract", input=str(out), output_dir=str(tmp_path / "dest"))["output_dir"]).name \
        == "pack (2)"


def test_archive_options_not_available(api, tmp_path):
    files = _files(tmp_path)
    err(api, ErrorCode.BACKEND_UNAVAILABLE, "archive_create", inputs=files, output=str(tmp_path / "a.zip"),
        password="pw")
    if A.compression_methods()["xz"] is None:
        err(api, ErrorCode.BACKEND_UNAVAILABLE, "archive_create", inputs=files, output=str(tmp_path / "b.zip"),
            compression="xz")
    err(api, ErrorCode.UNSUPPORTED_FORMAT, "archive_list", input=files[2])


def _zip(path: Path, entries: list[tuple[str, bytes]], symlink: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
        if symlink:
            zi = zipfile.ZipInfo(symlink)
            zi.external_attr = (0o120777 << 16)
            zf.writestr(zi, "/etc/passwd")
    return path


@pytest.mark.parametrize("evil", ["../evil.txt", "/abs/evil.txt", "C:/evil.txt", "a/../../evil.txt",
                                  "..\\evil.txt", "ok/./x.txt"])
def test_zip_slip_is_rejected_and_rolled_back(api, tmp_path, evil):
    z = _zip(tmp_path / "slip.zip", [("good.txt", b"fine"), (evil, b"pwned")])
    dest = tmp_path / "dest"
    dest.mkdir()
    err(api, ErrorCode.PATH_NOT_ALLOWED, "archive_extract", input=str(z), output_dir=str(dest))
    assert list(dest.iterdir()) == []  # the subfolder and good.txt were removed again
    assert not (tmp_path / "evil.txt").exists()
    lst = run(api, "archive_list", input=str(z))
    assert [e["path"] for e in lst["entries"]] == ["good.txt"] and lst["skipped_entries"][0]["path"] == evil


def test_safe_segments_port():
    assert A.safe_segments("a\\b//c/") == ["a", "b", "c"]
    assert A.safe_segments("x" * 300) == ["x" * 240]
    for bad in ("", "  ", "/a", "C:/a", "a/../b", "./a", "a\x00b", "/".join(["a"] * 257), "a" * 4097):
        assert A.safe_segments(bad) is None
    assert A.archive_name("Photos.tar.gz") == "Photos" and A.archive_name("x.cbz") == "x"
    assert A.archive_name(".zip") == "Archive" and A.archive_name("name.unknown") == "name"


def test_symlink_entries_are_skipped(api, tmp_path):
    z = _zip(tmp_path / "link.zip", [("a.txt", b"a")], symlink="link")
    x = run(api, "archive_extract", input=str(z), output_dir=str(tmp_path / "d"))
    assert x["extracted"] == 1 and x["skipped"] == [{"path": "link", "reason": "symlink"}]
    assert not (Path(x["output_dir"]) / "link").exists()


def test_zip_bomb_limits(api, tmp_path):
    z = _zip(tmp_path / "bomb.zip", [("zeros.bin", b"\0" * (8 * 1024 * 1024))])
    dest = tmp_path / "d"
    e = err(api, ErrorCode.RESOURCE_LIMIT, "archive_extract", input=str(z), output_dir=str(dest), max_ratio=100)
    assert "ratio" in e.message
    err(api, ErrorCode.RESOURCE_LIMIT, "archive_extract", input=str(z), output_dir=str(dest), max_ratio=0,
        max_total_mb=4)
    err(api, ErrorCode.RESOURCE_LIMIT, "archive_extract", input=str(z), output_dir=str(dest), max_ratio=0,
        max_entry_mb=1)
    assert not any(dest.rglob("*"))
    many = _zip(tmp_path / "many.zip", [(f"f{i}", b"") for i in range(20)])
    err(api, ErrorCode.RESOURCE_LIMIT, "archive_extract", input=str(many), output_dir=str(dest), max_entries=10)
    err(api, ErrorCode.RESOURCE_LIMIT, "archive_list", input=str(many), max_entries=10)


def test_lying_declared_size_is_caught(tmp_path):
    z = _zip(tmp_path / "lie.zip", [("big.bin", b"\0" * (3 * 1024 * 1024))])
    data = bytearray(z.read_bytes())
    # shrink the declared uncompressed size in the central directory and the local header
    cd = data.rfind(b"PK\x01\x02")
    struct.pack_into("<I", data, cd + 24, 10)
    struct.pack_into("<I", data, 22, 10)
    z.write_bytes(bytes(data))
    created: list[Path] = []
    root = tmp_path / "out"
    root.mkdir()
    with zipfile.ZipFile(z) as zf, pytest.raises((LazyRoomError, zipfile.BadZipFile)):
        A.extract(zf, root, password=None, preserve=True, skip_hidden=False, selected=None,
                  limits=A.Limits(max_total_bytes=1024 * 1024), check_write=lambda p: p, created=created)


def test_extraction_options(api, tmp_path):
    z = _zip(tmp_path / "opts.zip", [("dir/", b""), ("dir/a.txt", b"a"), ("dir/sub/b.txt", b"b"),
                                     (".hidden/c.txt", b"c"), ("dir/.d.txt", b"d"), ("e.txt", b"e")])
    x = run(api, "archive_extract", input=str(z), output_dir=str(tmp_path / "d1"), skip_hidden=True)
    assert sorted(Path(f).relative_to(x["output_dir"]).as_posix() for f in x["files"]) == \
        ["dir/a.txt", "dir/sub/b.txt", "e.txt"]
    x = run(api, "archive_extract", input=str(z), output_dir=str(tmp_path / "d2"), create_subfolder=False,
            preserve_structure=False, entries=["dir/sub/b.txt", "e.txt"])
    assert Path(x["output_dir"]) == tmp_path / "d2"
    assert sorted(Path(f).name for f in x["files"]) == ["b.txt", "e.txt"]


# --- ZipCrypto (traditional PKWARE encryption), written by hand because zipfile cannot encrypt


def _crc_table():
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        t.append(c)
    return t


_T = _crc_table()


class _ZipCrypto:
    def __init__(self, pw: bytes):
        self.k = [0x12345678, 0x23456789, 0x34567890]
        for b in pw:
            self._update(b)

    def _update(self, b: int):
        k0, k1, k2 = self.k
        k0 = (k0 >> 8) ^ _T[(k0 ^ b) & 0xFF]
        k1 = ((k1 + (k0 & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        k2 = (k2 >> 8) ^ _T[(k2 ^ (k1 >> 24)) & 0xFF]
        self.k = [k0, k1, k2]

    def encrypt(self, data: bytes) -> bytes:
        out = bytearray()
        for p in data:
            t = (self.k[2] | 2) & 0xFFFF
            out.append(p ^ (((t * (t ^ 1)) >> 8) & 0xFF))
            self._update(p)
        return bytes(out)


def _zipcrypto_zip(path: Path, name: str, data: bytes, pw: bytes, aes: bool = False) -> Path:
    crc = zlib.crc32(data)
    enc = _ZipCrypto(pw)
    body = enc.encrypt(bytes(range(11)) + bytes([crc >> 24])) + enc.encrypt(data)
    fn = name.encode()
    method = 99 if aes else 0
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 1, method, 0, 0x21, crc, len(body), len(data), len(fn), 0)
    cen = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 1, method, 0, 0x21, crc, len(body), len(data),
                      len(fn), 0, 0, 0, 0, 0o100644 << 16, 0)
    off = len(local) + len(fn) + len(body)
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(cen) + len(fn), off, 0)
    path.write_bytes(local + fn + body + cen + fn + end)
    return path


def test_zipcrypto_extraction(api, tmp_path):
    z = _zipcrypto_zip(tmp_path / "locked.zip", "secret.txt", b"top secret " * 50, b"opensesame")
    with zipfile.ZipFile(z) as zf:
        assert zf.read("secret.txt", pwd=b"opensesame") == b"top secret " * 50  # fixture sanity
    lst = run(api, "archive_list", input=str(z), password="wrong")
    assert lst["encryption"] == "password_required" and lst["entries"][0]["encrypted"] and lst["password_ok"] is False
    assert run(api, "archive_list", input=str(z), password="opensesame")["password_ok"] is True
    dest = tmp_path / "d"
    err(api, ErrorCode.INVALID_PARAMS, "archive_extract", input=str(z), output_dir=str(dest))
    e = err(api, ErrorCode.INVALID_PARAMS, "archive_extract", input=str(z), output_dir=str(dest), password="nope")
    assert "Wrong archive password" in e.message and "nope" not in json.dumps(e.to_dict())
    assert not any(dest.rglob("*"))
    x = run(api, "archive_extract", input=str(z), output_dir=str(dest), password="opensesame")
    assert Path(x["files"][0]).read_bytes() == b"top secret " * 50


def test_aes_zip_is_reported_unsupported(api, tmp_path):
    z = _zipcrypto_zip(tmp_path / "aes.zip", "a.txt", b"data", b"pw", aes=True)
    assert run(api, "archive_list", input=str(z))["encryption"] == "unsupported"
    err(api, ErrorCode.BACKEND_UNAVAILABLE, "archive_extract", input=str(z), output_dir=str(tmp_path / "d"),
        password="pw")


def test_archive_paths_go_through_the_policy(api, tmp_path):
    z = _zip(tmp_path / "p.zip", [("a.txt", b"a")])
    err(api, ErrorCode.PATH_NOT_ALLOWED, "archive_extract", input=str(z), output_dir="/usr/lib/lazyroom-test")
    err(api, ErrorCode.PATH_NOT_ALLOWED, "archive_create", inputs=[str(z)], output="/etc/x.zip")


# ===========================================================================
# Audio cover extractor (synthetic files built from the specifications)
# ===========================================================================

def _img_bytes(fmt: str, color=(200, 30, 60), size=(16, 12)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


PNG = _img_bytes("PNG")
JPG = _img_bytes("JPEG", (20, 160, 90), (24, 24))
MPEG_FRAMES = b"\xff\xfb\x90\x64" + bytes(413) + b"\xff\xfb\x90\x64" + bytes(413)


def _syncsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _unsynchronise(b: bytes) -> bytes:
    out = bytearray()
    for i, x in enumerate(b):
        out.append(x)
        if x == 0xFF and (i + 1 == len(b) or b[i + 1] == 0 or b[i + 1] >= 0xE0):
            out.append(0)
    return bytes(out)


def _apic(mime: str, ptype: int, desc: bytes, data: bytes, enc: int = 0) -> bytes:
    return bytes([enc]) + mime.encode() + b"\0" + bytes([ptype]) + desc + data


def id3v23(frames: list[tuple[bytes, bytes, int]], unsync: bool = False) -> bytes:
    body = b""
    for fid, payload, flags in frames:
        body += fid + struct.pack(">IH", len(payload), flags) + payload
    flags = 0
    if unsync:
        body, flags = _unsynchronise(body), 0x80
    return b"ID3\x03\x00" + bytes([flags]) + _syncsafe(len(body)) + body


def id3v24_unsync(payload: bytes) -> bytes:
    data = _syncsafe(len(payload)) + payload  # data length indicator
    data = _unsynchronise(data)
    frame = b"APIC" + _syncsafe(len(data)) + struct.pack(">H", 0x0003) + data
    ext = _syncsafe(6) + b"\x01\x00"  # minimal v2.4 extended header
    body = ext + frame + bytes(20)  # padding
    return b"ID3\x04\x00\x40" + _syncsafe(len(body)) + body


def id3v22(desc_utf16: str, data: bytes) -> bytes:
    payload = b"\x01" + b"PNG" + b"\x03" + desc_utf16.encode("utf-16") + b"\0\0" + data
    body = b"TT2" + len(b"\0abc").to_bytes(3, "big") + b"\0abc" + b"PIC" + len(payload).to_bytes(3, "big") + payload
    return b"ID3\x02\x00\x00" + _syncsafe(len(body)) + body


def flac_picture(data: bytes, mime: str = "image/png", ptype: int = 3, desc: str = "front") -> bytes:
    m, d = mime.encode(), desc.encode()
    return (struct.pack(">II", ptype, len(m)) + m + struct.pack(">I", len(d)) + d
            + struct.pack(">IIIII", 16, 12, 24, 0, len(data)) + data)


def flac_file(pic: bytes) -> bytes:
    streaminfo = bytes(34)
    return (b"fLaC" + bytes([0]) + len(streaminfo).to_bytes(3, "big") + streaminfo
            + bytes([0x80 | 6]) + len(pic).to_bytes(3, "big") + pic + b"\xff\xf8" + bytes(30))


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def m4a_file(data: bytes, kind: int = 14) -> bytes:
    covr = _box(b"covr", _box(b"data", struct.pack(">I", kind) + b"\0\0\0\0" + data))
    ilst = _box(b"ilst", _box(b"\xa9nam", _box(b"data", struct.pack(">I", 1) + bytes(4) + b"Song")) + covr)
    hdlr = _box(b"hdlr", bytes(8) + b"mdirappl" + bytes(9))
    meta = _box(b"meta", bytes(4) + hdlr + ilst)
    moov = _box(b"moov", _box(b"mvhd", bytes(100)) + _box(b"udta", meta))
    return _box(b"ftyp", b"M4A \0\0\0\0M4A mp42isom") + moov + _box(b"mdat", bytes(64))


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def _ogg_page(packet: bytes, seq: int, flags: int, serial: int = 0x1234) -> bytes:
    lacing = [255] * (len(packet) // 255) + [len(packet) % 255]
    hdr = b"OggS\0" + bytes([flags]) + struct.pack("<qIII", 0, serial, seq, 0) + bytes([len(lacing)]) + bytes(lacing)
    page = bytearray(hdr + packet)
    struct.pack_into("<I", page, 22, _ogg_crc(bytes(page)))
    return bytes(page)


def ogg_file(codec: str, comments: list[bytes]) -> bytes:
    vendor = b"LazyRoom test"
    tags = struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", len(comments))
    tags += b"".join(struct.pack("<I", len(c)) + c for c in comments)
    if codec == "vorbis":
        head = b"\x01vorbis" + bytes(23)
        com = b"\x03vorbis" + tags + b"\x01"
    else:
        head = b"OpusHead\x01\x02" + bytes(9)
        com = b"OpusTags" + tags
    return _ogg_page(head, 0, 2) + _ogg_page(com, 1, 0) + _ogg_page(b"\0" * 10, 2, 4)


def _pixels(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _ref(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


AUDIO_CASES = {
    "mp3_v23.mp3": lambda: id3v23([(b"TIT2", b"\0Song", 0), (b"APIC", _apic("image/png", 3, b"Cover\0", PNG), 0)])
    + MPEG_FRAMES,
    "mp3_v23_unsync.mp3": lambda: id3v23([(b"APIC", _apic("image/jpeg", 3, b"\0", JPG), 0)], unsync=True)
    + MPEG_FRAMES,
    "mp3_v23_zlib.mp3": lambda: id3v23([(b"APIC", struct.pack(">I", len(_apic("image/png", 3, b"\0", PNG)))
                                         + zlib.compress(_apic("image/png", 3, b"\0", PNG)), 0x0080)])
    + MPEG_FRAMES,
    "mp3_v24_unsync.mp3": lambda: id3v24_unsync(_apic("image/jpeg", 3, "Ä".encode("utf-8") + b"\0", JPG, enc=3))
    + MPEG_FRAMES,
    "mp3_v22.mp3": lambda: id3v22("Cover ✓", PNG) + MPEG_FRAMES,
    "song.flac": lambda: flac_file(flac_picture(PNG)),
    "id3_then.flac": lambda: id3v23([(b"TIT2", b"\0x", 0)]) + flac_file(flac_picture(JPG, "image/jpeg")),
    "song.m4a": lambda: m4a_file(PNG),
    "song.ogg": lambda: ogg_file("vorbis", [b"TITLE=x", b"METADATA_BLOCK_PICTURE="
                                            + base64.b64encode(flac_picture(PNG * 3))]),
    "song.opus": lambda: ogg_file("opus", [b"metadata_block_picture=" + base64.b64encode(flac_picture(JPG,
                                                                                                    "image/jpeg"))]),
}


@pytest.mark.parametrize("name", list(AUDIO_CASES))
def test_audio_cover_extraction(api, tmp_path, name):
    raw = AUDIO_CASES[name]()
    src = tmp_path / name
    src.write_bytes(raw)
    kind, pics = AC.read_pictures(src)
    assert len(pics) == 1
    expected = JPG if b"\xff\xd8\xff" == pics[0].data[:3] else PNG
    assert pics[0].data in (PNG, JPG, PNG * 3)
    r = run(api, "audio_cover_extract", input=str(src))
    assert Path(r["output"]["path"]).name == src.stem + ".png" and r["source"] == "embedded"
    assert np.array_equal(_pixels(r["output"]["path"]), _ref(expected))
    o = run(api, "audio_cover_extract", input=str(src), format="original", output=str(tmp_path / "raw.bin"))
    assert Path(o["output"]["path"]).read_bytes() == pics[0].data


def test_audio_cover_metadata_and_selection(tmp_path):
    raw = id3v23([(b"APIC", _apic("image/png", 0, b"other\0", PNG), 0),
                  (b"APIC", _apic("image/jpeg", 3, b"front\0", JPG), 0)]) + MPEG_FRAMES
    p = tmp_path / "two.mp3"
    p.write_bytes(raw)
    kind, pics = AC.read_pictures(p)
    assert kind == "mp3" and [x.describe()["type"] for x in pics] == ["other", "front_cover"]
    assert AC.choose(pics, None).data == JPG and AC.choose(pics, 0).data == PNG
    _, pics = AC.read_pictures(_write(tmp_path / "v22.mp3", id3v22("Cover ✓", PNG)))
    assert pics[0].description == "Cover ✓" and pics[0].mime == "image/png"
    _, pics = AC.read_pictures(_write(tmp_path / "s.m4a", m4a_file(JPG, 13)))
    assert pics[0].mime == "image/jpeg"


def _write(p: Path, data: bytes) -> Path:
    p.write_bytes(data)
    return p


def test_audio_cover_video_frame_fallback(api, tmp_path):
    import cv2

    clip = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(3):
        w.write(np.full((48, 64, 3), (0, 0, 255), np.uint8))
    w.release()
    if not clip.exists() or clip.stat().st_size == 0:
        pytest.skip("OpenCV cannot write MP4 here")
    r = run(api, "audio_cover_extract", input=str(clip))
    assert r["source"] == "video_frame" and r["pictures"] == []
    assert _pixels(r["output"]["path"])[10, 10, 0] > 240  # red frame (frameAtTime fallback like ImageToolbox)


def test_audio_without_cover(api, tmp_path):
    p = _write(tmp_path / "plain.mp3", id3v23([(b"TIT2", b"\0x", 0)]) + MPEG_FRAMES)
    e = err(api, ErrorCode.OPERATION_FAILED, "audio_cover_extract", input=str(p), video_frame_fallback=False)
    assert "No image" in e.message
    junk = _write(tmp_path / "junk.ogg", b"OggS" + bytes(40))
    with pytest.raises(LazyRoomError):
        run(api, "audio_cover_extract", input=str(junk), video_frame_fallback=False)


# ===========================================================================
# Code preview
# ===========================================================================

def _render(**kw):
    return CP.render(CP.CodePreviewParams(**kw))


def test_code_preview_geometry_matches_imagetoolbox_formulas():
    img, info = _render()
    font = CP.mono_font(45)
    asc, desc = font.getmetrics()
    lines = 8
    content = lines * (asc + desc) + (lines - 1) * int((asc + desc) * 0.55 + 0.5)
    card_h = math.ceil(126 + 3 + 120 + content)
    shadow = math.ceil((14 + 5) * 3)
    assert img.shape == (card_h + 2 * shadow + 2 * 84, 1680 + 2 * shadow + 2 * 84, 4)
    assert info["highlighter"] == ("pygments" if HAS_PYGMENTS else "none") and info["lines"] == 8
    again, _ = _render()
    assert np.array_equal(img, again)  # deterministic
    # rounded canvas corners are transparent, the gradient is opaque inside
    assert img[0, 0, 3] == 0 and img[img.shape[0] // 2, 2, 3] == 255


def test_code_preview_imagetoolbox_renderer_tests():
    no_pad, _ = _render(outer_padding=0, show_card_shadow=False)
    pad, _ = _render(outer_padding=8, show_card_shadow=False)
    assert (pad.shape[1] - no_pad.shape[1], pad.shape[0] - no_pad.shape[0]) == (48, 48)
    shadow, _ = _render(outer_padding=0, card_shadow_blur_radius=20, card_shadow_offset_x=30,
                        card_shadow_offset_y=-30)
    assert (shadow.shape[1] - no_pad.shape[1], shadow.shape[0] - no_pad.shape[0]) == (300, 300)
    empty, _ = _render(title="")
    placeholder, _ = _render(title="Untitled.kt")
    assert empty.shape == placeholder.shape and not np.array_equal(empty, placeholder)
    p = CP.CodePreviewParams(rotation=12)
    preview, _ = CP.render(p, max_pixels=2_000_000)
    export, _ = CP.render(p)
    assert preview.shape[0] * preview.shape[1] <= 2_000_000
    assert abs(preview.shape[1] / preview.shape[0] - export.shape[1] / export.shape[0]) < 0.01


def test_code_preview_long_code_stays_inside_bitmap_limits():
    code = "\n".join(f"val renderedLine{i} = renderCodePreview(line = {i})" for i in range(1, 1201))
    img, info = _render(code=code, rotation=15, show_canvas_background=False)
    assert img.shape[0] > 4096 and img.shape[0] * img.shape[1] <= 65_545_216
    assert img[0, 0, 3] == 0 and info["output_scale"] < 1


def test_code_preview_themes_and_languages():
    from lazyroom.tools._code_themes import THEMES

    assert len(THEMES) == 256
    for name in THEMES:
        t = CP.get_theme(name)
        assert len(t.background) == 4 and len(t.text) == 4 and len(t.scopes) == len(CP.SCOPES)
    assert {"atom-one-dark", "github-dark", "night-owl", "base16/dracula", "base16/material-palenight",
            "base16/solarized-light", "base16/windows-95"} <= set(THEMES)
    assert CP.theme_title("base16/dracula") == "Dracula" and CP.theme_title("base16/tomorrow-night") == \
        "Tomorrow Night" and CP.theme_title("vs2015") == "VS 2015"
    assert len(CP.LANGUAGE_KEYS) == 192 and CP.LANGUAGE_KEYS[:3] == ["kotlin", "java", "javascript"]
    assert CP.language_title("cpp") == "C++" and CP.language_title("clojure-repl") == "Clojure REPL"
    assert CP.language_title("node-repl") == "Node.js REPL" and CP.language_extension("kotlin") == "kt"
    dracula = CP.get_theme("base16/dracula")
    assert dracula.background[:3] == (0x28, 0x29, 0x36) and dracula.scopes["keyword"][:3] == (0xB4, 0x5B, 0xCF)
    if not HAS_PYGMENTS:  # pragma: no cover
        pytest.skip("pygments not installed")
    spans, engine = CP.highlight(CP.DEFAULT_CODE, "kotlin", dracula)
    assert engine == "pygments" and (0, 3, dracula.scopes["keyword"]) in spans
    assert CP.highlight("x", "taggerscript", dracula) == ([], "none")


def test_code_preview_without_pygments(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygments.lexers", None)
    assert CP.highlight(CP.DEFAULT_CODE, "kotlin", CP.get_theme("github")) == ([], "none")
    img, info = _render(code="x = 1")
    assert info["highlighter"] == "none" and img.shape[2] == 4


def test_code_preview_wrapping_and_tabs():
    font = CP.mono_font(45)
    cw = font.getlength("a")
    lines = CP.layout_lines("aaaa bbbb cccc\nx", font, cw * 10)
    assert lines == [(0, 10), (10, 14), (15, 16)]  # break after the space, trailing space hangs
    assert CP.layout_lines("a" * 25, font, cw * 10) == [(0, 10), (10, 20), (20, 25)]
    assert CP._advance(font, "\t", 0.0) == 20.0 and CP._advance(font, "\t", 20.0) == 20.0  # Android tab stops


def test_code_preview_tool(api, tmp_path):
    src = tmp_path / "hello.py"
    src.write_text("def hello():\n    return 'world'\n")
    r = run(api, "code_preview", input=str(src), language="python", theme="github-dark", background="sunset",
            title="hello.py", output=str(tmp_path / "out.png"))
    assert r["output"]["width"] == r["width"] and r["lines"] == 3
    with Image.open(r["output"]["path"]) as im:
        assert im.mode == "RGBA" and im.size == (r["width"], r["height"])
    c = run(api, "code_preview", background="custom", background_colors=["#ff0000", "#00ff00", "#0000ff"],
            output=str(tmp_path / "c.png"))
    assert c["output"]["path"].endswith("c.png")
    err(api, ErrorCode.INVALID_PARAMS, "code_preview", background="custom", output=str(tmp_path / "d.png"))


# ===========================================================================
# Document scanner
# ===========================================================================
PAGE_W, PAGE_H = 850, 1100


def _page() -> np.ndarray:
    rng = np.random.default_rng(1)
    page = np.full((PAGE_H, PAGE_W, 3), 245, np.uint8)
    for i in range(18):
        y = 120 + i * 50
        page[y:y + 18, 100:120 + int(rng.integers(300, 600))] = 30
    page[40:80, 100:700] = (180, 40, 40)
    return page


def _photo(corners, w=1200, h=900, shading=True) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(0)
    bg = np.clip(np.array([70, 55, 45], np.float32) + rng.normal(0, 12, (h, w, 3)), 0, 255)
    src = np.float32([[0, 0], [PAGE_W - 1, 0], [PAGE_W - 1, PAGE_H - 1], [0, PAGE_H - 1]])
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    warped = cv2.warpPerspective(_page(), M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
    mask = cv2.warpPerspective(np.ones((PAGE_H, PAGE_W), np.float32), M, (w, h))[..., None]
    out = warped * mask + bg * (1 - mask)
    if shading:
        out *= (0.65 + 0.35 * np.arange(w, dtype=np.float32) / w)[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


TRUE_CORNERS = [[260, 90], [930, 140], [1010, 820], [170, 760]]


def test_document_page_detection_and_rectification():
    import cv2

    photo = _photo(TRUE_CORNERS)
    corners, info = D.detect_page(photo)
    assert info["detected"]
    assert D.corner_error(corners, np.array(TRUE_CORNERS, float)) < 0.005 * D.diag(*photo.shape[:2])
    flat = D.rectify(photo, corners)
    page = cv2.resize(_page(), (flat.shape[1], flat.shape[0]), interpolation=cv2.INTER_AREA)
    clean = D.enhance(flat, "shadow_removal")
    assert np.mean(np.abs(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(float)
                          - cv2.cvtColor(page, cv2.COLOR_RGB2GRAY))) < 20
    bw = D.enhance(flat, "bw")
    assert bw.ndim == 2 and set(np.unique(bw)) <= {0, 255}
    text = cv2.resize(_page(), (bw.shape[1], bw.shape[0]))[..., 0] < 100
    assert np.mean((bw == 0) == text) > 0.95
    assert D.enhance(flat, "grayscale").ndim == 2 and D.enhance(flat, "color") is flat


def test_document_scan_tool_images_and_pdf(api, tmp_path):
    paths = []
    for i, corners in enumerate((TRUE_CORNERS, [[200, 60], [1000, 60], [1000, 860], [200, 860]])):
        p = tmp_path / f"page{i}.png"
        save_image(_photo(corners).astype(np.float32) / 255, p)
        paths.append(str(p))
    r = run(api, "document_scan", inputs=paths, mode="bw", output=str(tmp_path / "scans"), format="png")
    assert [Path(pg["output"]["path"]).name for pg in r["pages"]] == ["page0_scan.png", "page1_scan.png"]
    assert all(pg["detected"] for pg in r["pages"])
    pdf = run(api, "document_scan", inputs=paths, mode="shadow_removal", format="pdf",
              output=str(tmp_path / "doc.pdf"))
    assert pdf["output"]["pages"] == 2
    from pypdf import PdfReader

    assert len(PdfReader(pdf["output"]["path"]).pages) == 2
    user = run(api, "document_scan", inputs=paths[:1], corners=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
               output=str(tmp_path / "u"))
    assert user["pages"][0]["method"] == "user" and user["pages"][0]["corners"][0] == [0.1, 0.1]
    err(api, ErrorCode.INVALID_PARAMS, "document_scan", inputs=paths, corners=[[0, 0], [1, 0], [1, 1], [0, 1]])
    noise = tmp_path / "noise.png"
    save_image(np.random.default_rng(3).random((300, 400, 3)).astype(np.float32), noise)
    n = run(api, "document_scan", inputs=[str(noise)], output=str(tmp_path / "n"))
    assert n["pages"][0]["detected"] is False and "warning" in n["pages"][0]


# ===========================================================================
# Screenshot framing: already ported in tools/itb_tools.py (another module) -> no duplicate tool here
# ===========================================================================

def test_no_tool_name_collision_with_itb_tools(api, tmp_path):
    try:
        import lazyroom.tools.itb_tools  # noqa: F401
    except Exception as exc:  # pragma: no cover - other module under construction
        pytest.skip(f"itb_tools not importable: {exc}")
    assert TOOLS["screenshot_framing"].fn.__module__ == "lazyroom.tools.itb_tools"
    # ImageToolbox ScreenshotFramingBitmapRendererTest: geometry of the framed result
    src = tmp_path / "shot.png"
    save_image(np.ones((240, 100, 3), np.float32) * [1, 0, 0], src)
    plain = dict(input=str(src), preset="custom", start_color="#FFFFFF", end_color="#FFFFFF", gradient=False,
                 padding=0, corner_radius=0, shadow=False)
    r = run(api, "screenshot_framing", output=str(tmp_path / "a.png"), **plain)
    assert (r["output"]["width"], r["output"]["height"]) == (100, 240)
    r = run(api, "screenshot_framing", output=str(tmp_path / "b.png"), aspect_ratio="16:9", **plain)
    assert abs(r["output"]["width"] / r["output"]["height"] - 16 / 9) < 0.01
