# Portions of this file are covered by the MIT License
# Copyright (c) 2024 Joshua Carroll
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import uvicorn

import os
from LegalDefAgent.src.settings import settings


if __name__ == "__main__":
    uvicorn.run("LegalDefAgent.agent_service_toolkit.src.service.service:app", host=settings.HOST, port=settings.PORT, reload=settings.is_dev())
