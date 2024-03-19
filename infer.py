import os
import sys

from logtool.logtool import log

@log("info")
def test(s: str):
    print(s)

if __name__ == "__main__":
    test('a')
