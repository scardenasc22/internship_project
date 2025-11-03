from langgraph.graph import StateGraph, END, START
from schemas import WorkflowState
from nodes import (
    criteria_generation, 
    llm_criteria_evaluation,
    route_evalution,
    human_criteria_assessment,
    get_company_info,
    get_candidates_info,
    score_candidates,
    export_scores,
    generate_questions,
    calculate_overall_score,
    extract_experience,
    candidates_tournament,
    tournament_simulation,
    generate_candidate_overview
    # generate_explanations
)
from functions import text_extraction
import os

# creating the graph
workflow = StateGraph(WorkflowState)
# adding nodes
workflow.add_node("criteria_generation", criteria_generation)
workflow.add_node("llm_criteria_evaluation", llm_criteria_evaluation)
workflow.add_node("human_criteria_assessment", human_criteria_assessment)
workflow.add_node("get_company_info", get_company_info)
workflow.add_node("get_candidates_info", get_candidates_info)
workflow.add_node("score_candidates", score_candidates)
workflow.add_node("candidates_tournament", candidates_tournament)
workflow.add_node("export_scores", export_scores)
workflow.add_node("calculate_overall_score", calculate_overall_score)
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("extract_experience", extract_experience)
workflow.add_node("tournament_simulation", tournament_simulation)
workflow.add_node("generate_candidate_overview", generate_candidate_overview)

# workflow.add_node("generate_explanations", generate_explanations)
# connecting the nodes
workflow.add_edge(START, "get_company_info")
workflow.add_edge("get_company_info", "criteria_generation")
workflow.add_edge("criteria_generation", "llm_criteria_evaluation")
# conditional nodes
workflow.add_conditional_edges(
    source = "llm_criteria_evaluation",
    path = route_evalution,
    path_map = {
        "accepted" : "human_criteria_assessment",
        "rejected" : "criteria_generation"
    }
)
workflow.add_conditional_edges(
    source = "human_criteria_assessment",
    path = route_evalution,
    path_map = {
        "accepted" : "get_candidates_info",
        "rejected" : "criteria_generation"
    }
)
# connecting nodes after decision
workflow.add_edge("get_candidates_info", "tournament_simulation")
workflow.add_edge("tournament_simulation", "candidates_tournament")
workflow.add_edge("candidates_tournament", "score_candidates")
workflow.add_edge("score_candidates", "export_scores")
workflow.add_edge("export_scores", "calculate_overall_score")
workflow.add_edge("calculate_overall_score", "extract_experience")
workflow.add_edge("extract_experience", "generate_questions")
workflow.add_edge("generate_questions", "generate_candidate_overview")
workflow.add_edge("generate_candidate_overview", END)
# workflow.add_edge("generate_explanations", END)
# compile the workflow
compiled_workflow = workflow.compile()

### testing the workflow
# sample input
root = os.getcwd()

test_input = WorkflowState(
    job_description = text_extraction(
        file_path = os.path.join(root, "data/raw/job/data_role_des.txt")
    ),
    candidates_folder = os.path.join(root, "data", "raw", "cv"),
    count = 0,
    scores_folder = os.path.join(root, "data", "gpt_5_results"),
    batch_size = 5,
    selected_per_batch = 2
)
# test the workflow
compiled_workflow.invoke(
    input = test_input
)