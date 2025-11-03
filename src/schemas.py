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

class QAItem(BaseModel):
    question: str = Field(description="Interview question")
    look_for: str = Field(description="What the recruiter should look for in the candidate's answer")

    @field_validator("question", "look_for")
    @classmethod
    def must_be_non_empty_trimmed(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Value cannot be empty")
        return v

class InterviewQuestions(BaseModel):
    """generation of interview questions"""
    experience_questions: List[QAItem] = Field(
        description="4 items clarifying or expanding upon the candidate's experience"
    )
    situational_questions: List[QAItem] = Field(
        description="4 items about behavior, problem-solving, and soft skills"
    )
    technical_questions: List[QAItem] = Field(
        description="4 items about tools, technologies, and practices relevant to the role"
    )

    @field_validator("experience_questions", "situational_questions", "technical_questions")
    @classmethod
    def exactly_four(cls, v: List[QAItem]) -> List[QAItem]:
        if len(v) != 4:
            raise ValueError("Each category must contain exactly 4 items")
        return v

class JobExperience(BaseModel):
    """Details about a single job experience."""
    job_title: str = Field(..., description="The candidate's job title.")
    company_name: str = Field(..., description="The name of the company.")
    years_of_experience: str = Field(..., description="The duration of employment (e.g., '2020 - 2023' or '2 years').")
    responsibilities: List[str] = Field(..., description="A list of key responsibilities and accomplishments.")

class CandidateExperience(BaseModel):
    """The structured extraction of a candidate's professional experience."""
    experience: List[JobExperience] = Field(..., description="A list of all professional experiences listed on the resume.")

class CandidateRationale(BaseModel):
    """The rationale for a single candidate's score."""
    candidate_id: str = Field(..., description="The unique ID of the candidate.")
    rationale: List[str] = Field(..., description="A concise explanation of the candidate's score in the specified criterion, in comparison to other candidates in the group.")

class GroupRationales(BaseModel):
    """A list of rationales for a group of candidates."""
    rationales: List[CandidateRationale] = Field(..., description="A list of rationales for each candidate in the group.")

class WinnerRationale(BaseModel):
    """Details for a single selected candidate."""
    candidate_id: str = Field(..., description="The unique ID of the selected winner.")
    justification: str = Field(..., description="A 1-2 sentence explanation citing specific evidence from the resume that qualifies this candidate over the others.")

class GroupWinners(BaseModel):
    """The structured output containing all selected winners and their rationale."""
    selected_winners: List[WinnerRationale] = Field(..., description="A list of 1 or 2 candidates selected as the best in the group, each with a justification.")
    # You can also add a field for an overall comparison summary if helpful
    overall_summary: str = Field(..., description="A brief summary of the key difference between the winners and the rest of the group.")

class CandidateFeedback(BaseModel):
    overview : str = Field(
        description = "A comprehensive summary of the candidate based on their resume, highlighting their background, experience, and overall fit for the role"
    )
    strengths : List[str] = Field(
        description = "list of the candidate's strengths for the target role"
    )
    weaknesses : List[str] = Field(
        description = "list of the candidate's weaknesses for the target role"
    )
    @field_validator("strengths", "weaknesses")
    @classmethod
    def exactly_three(cls, v: List[str]) -> List[str]:
        if len(v) != 3:
            raise ValueError("Each category must contain exactly 3 items")
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
    # dictionary of the scores of each candidate
    candidates_scores : Optional[Dict] = None
    # tournament conditions
    batch_size : Optional[int] = 5
    selected_per_batch : Optional[int] = 2
    number_of_rounds : Optional[int] = 2
    # storing folder
    scores_folder : str 
    # dictionary with candidate experience
    candidates_exp : Optional[Dict] = None
    # candidates names dictionary id : name pairs (edited files)
    candidates_names : Optional[Dict] = None
    # selection rationale during the tournament
    round_rationale : Optional[Dict] = {}
    # overall rationale per round
    round_summaries : Optional[Dict] = {}
    # initial groups dictionary
    initial_groups : Optional[Dict] = {}
    # candidate tournament winners
    top_candidates : Optional[Dict] = {}