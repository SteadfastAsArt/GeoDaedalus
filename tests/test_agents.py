# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# @Project ：GeoDaedalus
# @Author ：qm
# @Date ：2025/6/11 15:34
# @Description:

from uuid import UUID, uuid4
from geodaedalus.core.config import get_settings, Settings
from geodaedalus.agents import (
    RequirementUnderstandingAgent,
    LiteratureSearchAgent,
)
from geodaedalus.core.models import (
    GeoscientificConstraints,
    LiteraturePaper,
    SearchResults,
    RockType,
    ElementConstraints,
    ElementCategory
)


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
                    elements=["SiO2", "Al2O3", "Fe2O3"],
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


class TestDataExtractionAgent:
    def test_initialize(self):
        pass


class TestDataFusionAgent:
    def test_initialize(self):
        pass
