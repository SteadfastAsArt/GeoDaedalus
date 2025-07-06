# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# @Project: GeoDaedalus
# @Author: qm
# @Date: 2025/6/20 12:59
# @Description:


import asyncio
import os.path
from uuid import UUID, uuid4
from pydantic import BaseModel
from typing import Type, TypeVar
from geodaedalus.core.config import get_settings, Settings
from geodaedalus.agents import (
    RequirementUnderstandingAgent,
    LiteratureSearchAgent,
    DataExtractionAgent
)
from geodaedalus.core.models import (
    GeoscientificConstraints,
    LiteraturePaper,
    SearchResults,
    RockType,
    ElementConstraints,
    ElementCategory,
    ExtractedData
)

T = TypeVar("T", bound=BaseModel)


def save_model(model: BaseModel, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))


def load_model(model_cls: Type[T], filepath: str) -> T:
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
    return model_cls.model_validate_json(data)


async def main():
    suffix = 8
    search_result_file_path = f"./temp/search_results/search_results_{suffix}.json"
    suffix_1 = 1
    extracted_data_file_path = f"./temp/extracted_data/extracted_data_{suffix}_{suffix_1}.json"
    while os.path.exists(extracted_data_file_path):
        suffix_1 += 1
        extracted_data_file_path = f"./temp/extracted_data/extracted_data_{suffix}_{suffix_1}.json"
    print("Loading search results ...")
    test_search_results: SearchResults = load_model(SearchResults, search_result_file_path)
    # print(f"===== Loaded search results ====\n{test_search_results}")
    print("Initializing DataExtractionAgent ...")
    agent = DataExtractionAgent(
        session_id=uuid4(),
        settings=get_settings()
    )
    print("Extracting data from search results ...")
    result = await agent.process(
        search_results=test_search_results,
        max_papers=-1,
    )
    assert result is not None
    assert isinstance(result, ExtractedData)
    print(f"===== Extracted Data ====\n{result}")
    save_model(result, extracted_data_file_path)


if __name__ == "__main__":
    asyncio.run(main())
