from pydantic import BaseModel, Field, field_validator
from typing import List, Any, Optional, Literal, Dict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, AIMessage
from numpy import sum

class EvaluationCriteria(BaseModel):
    """criteria to evaluate candidates"""
    domains : List[str] = Field(
        description = "List of key subject areas or industries relevant to the role"
    )
    technical_skills : List[str] = Field(
        description = "List of specific technologies, programming languages, and tools required."
    )
    soft_skills : List[str] = Field(
        description = "List of interpersonal abilities and qualities necessary for success at work"
    )
    culture : List[str] = Field(
        description = "List of traits that align with the company's culture and values"
    )
    weights : List[int] = Field(
        description = "List of weights for each component"
    )
    @field_validator("weights")
    @classmethod
    def check_weigths(cls, v : List[int]):
        if sum(v) != 100:
            raise ValueError("The weights must add to exactly 100")
        return v
    @field_validator("domains", "technical_skills", "soft_skills", "culture")
    @classmethod
    def check_criteria(cls, v: List[str]):
        if len(v) == 0:
            raise ValueError("The list should contain at least one value")
        return v

class EvaluationScores(BaseModel):
    """A flat list of scores for each evaluation criterion."""
    scores: List[int] = Field(
        ...,
        description = "A list of scores from 0 to 100 for each evaluation criteria point, in the exact order they were provided in the prompt."
    )

    @field_validator("scores")
    @classmethod
    def check_scores_and_length(cls, v: List[int]):
        # This validator is crucial for ensuring data integrity
        if any(not 0 <= score <= 100 for score in v):
            raise ValueError("All scores must be between 0 and 100.")
        return v

class Refinement(BaseModel):
    """schema for LLM feedback"""
    grade : Literal["good", "needs improvement"] = Field(
        description = "Grade on the criteria"
    )
    feedback : str = Field(
        description = "If the criteria needs improvement, provide the specific feedback"
    )

class CompanyDetails(BaseModel):
    """information related to the company"""
    name : str = Field(
        ...,
        description = "Name of the company"
    )
    location : Optional[str] = Field(
        None,
        description = "The company's location if specified"
    )
    sector : Optional[str] = Field(
        None,
        description = "The industry or sector the company belongs to"
    )
    description : str = Field(
        ...,
        description = "Brief description of the company"
    )

class InterviewQuestions(BaseModel):
    """generation of interview questions"""
    name : str = Field(
        description = "Name of the candidate"
    )
    experience_questions : List[str] = Field(
        description = "List of questions aimed to clarify or expand upon the candidate's professional experience"
    )
    situational_questions : List[str] = Field(
        description = "List of questions aimed to understand the candidate's behavior, problem-solving abilities, and soft skills in relation to the target role"
    )
    @field_validator("experience_questions", "situational_questions")
    @classmethod
    def check_questions(cls, v: List[int]):
        # This validator is crucial for ensuring data integrity
        if len(v) == 0:
            raise ValueError("The list should contain at least one value")
        return v

class WorkflowState(BaseModel):
    """information that will flow through the graph"""
    messages : Annotated[List[AnyMessage], add_messages] = None
    job_criteria : Optional[EvaluationCriteria] = None
    company_info : Optional[CompanyDetails] = None
    job_description : str
    criteria_status : Optional[Literal["good", "needs improvement"]] = None
    criteria_feedback : Optional[str] = None
    candidates_folder : str
    count : Optional[int] = 0
    # dictionary with the "id" : "resume" pairs
    candidates_dict : Optional[Dict[str, str]] = None
    # dictory of the scores of each candidate
    candidates_scores : Optional[Dict] = None
    # number of batches
    batches : Optional[int] = 4
    # storing folder
    scores_folder : str 