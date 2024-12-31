from typing import Literal


class Config:
    backend: Literal["py", "rs"] = "py"


config = Config()
