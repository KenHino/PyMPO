from typing import Literal


class Config:
    backend: Literal["py", "rs"] = "rs"


config = Config()
