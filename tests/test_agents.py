# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# @Project ：GeoDaedalus
# @Author ：qm
# @Date ：2025/6/11 15:34
# @Description:

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


class TestBaseAgent:
    """
    The BaseAgent class contains the process method decorated with @abstractmethod,
    so it is an abstract class and cannot be directly instantiated.
    Only subclasses that have implemented all the abstract methods can be instantiated.
    """
    pass


class TestRequirementUnderstandingAgent:
    def test_initialize(self):
        agent = RequirementUnderstandingAgent(
            session_id=uuid4(),
            settings=get_settings()
        )
        assert agent.session_id is not None
        assert isinstance(agent.session_id, UUID)
        print(agent.session_id)
        assert agent.settings is not None
        assert isinstance(agent.settings, Settings)
        print(agent.settings)


class TestLiteratureSearchAgent:
    def test_initialize(self):
        agent = LiteratureSearchAgent(
            session_id=uuid4(),
            settings=get_settings()
        )
        assert agent.session_id is not None
        assert isinstance(agent.session_id, UUID)
        print(agent.session_id)
        assert agent.settings is not None
        assert isinstance(agent.settings, Settings)
        print(agent.settings)

    async def test_process(self):
        agent = LiteratureSearchAgent(
            session_id=uuid4(),
            settings=get_settings()
        )
        constraints = GeoscientificConstraints(
            spatial=None,
            temporal=None,
            rock_types=[RockType.VOLCANIC, RockType.CLASTIC, RockType.CARBONATE],
            element_constraints=[
                ElementConstraints(
                    category=ElementCategory.MAJOR,
                    elements=["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O"],
                    required_elements=None,
                    excluded_elements=None
                )
            ],
            additional_keywords=[],
        )
        result = await agent.process(constraints)
        assert result is not None
        assert isinstance(result, SearchResults)
        # print(result)
        for idx, paper in enumerate(result.papers):
            if idx == 0:
                assert isinstance(paper, LiteraturePaper)
            print(f"Paper {idx + 1}:\n{paper.title}\n{paper.web_url}")

        suffix = 1
        search_result_file_path = f"./temp/search_results/search_results_{suffix}.json"
        while os.path.exists(search_result_file_path):
            suffix += 1
            search_result_file_path = f"./temp/search_results/search_results_{suffix}.json"
        save_model(result, search_result_file_path)


class TestDataExtractionAgent:
    def test_initialize(self):
        agent = DataExtractionAgent(
            session_id=uuid4(),
            settings=get_settings()
        )
        assert agent.session_id is not None
        assert isinstance(agent.session_id, UUID)
        print(agent.session_id)
        assert agent.settings is not None
        assert isinstance(agent.settings, Settings)
        print(agent.settings)

    async def test_process(self):
        suffix = 8
        search_result_file_path = f"./temp/search_results/search_results_{suffix}.json"
        suffix_1 = 1
        extracted_data_file_path = f"./temp/extracted_data/extracted_data_{suffix}_{suffix_1}.json"
        while os.path.exists(extracted_data_file_path):
            suffix_1 += 1
            extracted_data_file_path = f"./temp/extracted_data/extracted_data_{suffix}_{suffix_1}.json"
        test_search_results: SearchResults = load_model(SearchResults, search_result_file_path)
        print(f"===== Loaded search results ====\n{test_search_results}")
        agent = DataExtractionAgent(
            session_id=uuid4(),
            settings=get_settings()
        )
        result = await agent.process(
            search_results=test_search_results,
            max_papers=11,
        )
        assert result is not None
        assert isinstance(result, ExtractedData)
        print(f"===== Extracted Data ====\n{result}")
        save_model(result, extracted_data_file_path)


class TestDataFusionAgent:
    def test_initialize(self):
        pass
