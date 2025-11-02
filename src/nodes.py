from math import ceil
from schemas import (
    Refinement, 
    WorkflowState, 
    EvaluationCriteria,
    CompanyDetails,
    EvaluationScores,
    InterviewQuestions,
    CandidateExperience,
    GroupRationales,
    GroupWinners
)
from functions import (
    print_execution_status,
    text_extraction,
    scores_to_dataframe,
    auxiliary_col_calutation,
    refined_overall_calculation,
    execution_time,
    format_experience_for_prompt,
    candidate_simulation
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from prompts import (
    criteria_prompt, 
    criteria_with_feedback_prompt, 
    refinement_prompt, 
    company_details_request_prompt,
    scoring_prompt,
    interview_questions_prompt,
    exp_extraction_prompt,
    rationale_prompt,
    selection_prompt
)
from langchain_core.messages import HumanMessage, AIMessage
from typing import Any, Literal
from pandas import read_csv
from json import dump
import os
import numpy as np
from itertools import combinations
from collections import defaultdict
from tqdm.auto import tqdm
import random
random.seed(2000)

# loading environment variables
load_dotenv()

# one of the multiple LLMs that can be leveraged according to the task
llm = ChatOpenAI(
    model = "gpt-4o-2024-08-06"
)
# tools to search information on the web
web_search = TavilySearch(
    max_results = 3
)

### nodes
# 1. company details extraction
@print_execution_status
def get_company_info(state : WorkflowState) -> WorkflowState:
    """the agent extracts the company details"""
    # create the chain of response
    llm_extractor = llm.with_structured_output(schema = CompanyDetails)
    info_extraction_chain = company_details_request_prompt | llm_extractor
    # extract info from the job description
    initial_details = info_extraction_chain.invoke(
        {
            "job_description" : state.job_description
        }
    )
    # update the state of considering the extracted details
    state.company_info = initial_details
    # prepare the web search
    search_query = f"{initial_details.name}, {initial_details.sector}, profile"
    web_results = web_search.invoke({
        "query" : search_query
    })
    # extract the content of the websearch
    details = [f"Source {i + 1}: {info['content']}" for i, info in enumerate(web_results['results'])]
    # adding the internet search to the company description
    state.company_info.description += (
        f"Information from the internet: "
        "\n".join(details)
    )
    return state
    
# 2. criteria generation node
@print_execution_status
def criteria_generation(state : WorkflowState) -> WorkflowState:
    """the agent generates the criteria to evaluate each candidate"""
    llm_constrained = llm.with_structured_output(schema = EvaluationCriteria)
    # check if there is any feedback from previous interactions
    if not state.criteria_feedback: # initial generation when there is no feedback
        initial_messages = criteria_prompt.invoke({
            "job_description" : state.job_description,
            "company_info" : (
                f"Company Name: {state.company_info.name}\n"
                f"Location: {state.company_info.location}\n"
                f"Sector: {state.company_info.sector}\n"
                f"Description: {state.company_info.description}\n"
            )
        }).messages
        state.messages = initial_messages
    else:
        followup_messages = criteria_with_feedback_prompt.invoke({
            "job_description" : state.job_description,
            "feedback" : state.criteria_feedback,
            "company_info" : (
                f"Company Name: {state.company_info.name}\n"
                f"Location: {state.company_info.location}\n"
                f"Sector: {state.company_info.sector}\n"
                f"Description: {state.company_info.description}\n"
            )
        }).messages
        state.messages += followup_messages
    msg = llm_constrained.invoke(input = state.messages) # structured response from the llm
    state.job_criteria = msg # update the criteria of the state graph
    # append message of the generated criteria
    state.messages += [
        AIMessage(content = (
            f"The generated criteria for the job description provided is:\n"
            f"Domains: {','.join(state.job_criteria.domains)}\n"
            f"Technical skills: {','.join(state.job_criteria.technical_skills)}\n"
            f"Soft skills: {','.join(state.job_criteria.soft_skills)}\n"
            f"Culture: {','.join(state.job_criteria.culture)}"
        ))
    ]
    return state

# 3. feedback from another LLM
@print_execution_status
def llm_criteria_evaluation(state : WorkflowState) -> WorkflowState:
    """the agent evaluates the criteria for the role"""
    llm_constrained = llm.with_structured_output(schema = Refinement)
    # concatenating the whole criteria into a single string
    criteria_string = (
        f"Domains: {state.job_criteria.domains}\n"
        f"Technical skills: {state.job_criteria.technical_skills}\n"
        f"Soft skills: {state.job_criteria.soft_skills}\n"
        f"Culture: {state.job_criteria.culture}\n"
    )
    # refinement messages
    refinement_messages = refinement_prompt.invoke({
        "job_description" : state.job_description,
        "criteria" : criteria_string
    }).messages
    # update the messages
    state.messages += refinement_messages
    # getting the structured response
    msg = llm_constrained.invoke(input = state.messages)
    # updating the state of the graph
    state.criteria_feedback = msg.feedback
    state.criteria_status = msg.grade
    state.count += 1
    # append the feedback to the conversation
    state.messages += [AIMessage(content = (
        f"The feedback on the criteria is: {state.criteria_feedback}"
    ))]
    return state

# 4. human feedback node
@print_execution_status
def human_criteria_assessment(state : WorkflowState) -> WorkflowState:
    """Prompts the user to provide feedback on the listed criteria"""
    criteria_string = (
        f"Domains: {', '.join(state.job_criteria.domains)}\n"
        f"Technical skills: {', '.join(state.job_criteria.technical_skills)}\n"
        f"Soft skills: {', '.join(state.job_criteria.soft_skills)}\n"
        f"Culture: {', '.join(state.job_criteria.culture)}\n"
        f"Weigths: {state.job_criteria.weights}\n"
    )
    print(f"The job evaluation criteria is the following:")
    print("*"*30)
    print(f"{criteria_string}")
    print("*"*30)
    human_feedback = input("Do you approve this criteria? (yes/no). If not, provide feedback: ")
    if human_feedback.lower().strip() == "yes":
        print(f"Approved criteria!")
        # flush the feedback and change workflow state
        state.criteria_feedback = ""
        state.criteria_status = "good"
        state.count = 0 # we might need that later
    else:
        print(f"Looping back for refinement")
        state.criteria_feedback = human_feedback
        state.criteria_status = "needs improvement"
        # update the messages
        state.messages += [HumanMessage(content = (
            f"The feedback on the criteria is: {state.criteria_feedback}"
        ))]
        state.count = 0
    return state

# 5. Collecting candidates info
@print_execution_status
def get_candidates_info(state : WorkflowState) -> WorkflowState:
    """Populate the information from each of the candidates in a dictionary"""
    try:
        if len(os.listdir(state.candidates_folder)) == 0:
            raise ValueError(f"the folder path: '{state.candidates_folder}' is empty")
        resumes_list = os.listdir(state.candidates_folder)
        state.candidates_dict = {
            file.split("-")[0] : text_extraction(os.path.join(state.candidates_folder, file)) for file in resumes_list
        }
        state.candidates_names = {
            file.split("-")[0] : file.split(sep = '.')[0].split(sep = "-")[1].replace("_", " ").title() for file in resumes_list
        }
    except Exception as e:
        raise Exception(f"Error while populating candidates info: {e}")
    return state

# 6. candidates tournament simulation
def tournament_simulation(state : WorkflowState) -> WorkflowState:
    """asks for user feedback about the parameters of the tournament and how many rounds to run"""
    population = len(state.candidates_dict.keys())
    parameters_approval = False
    while not parameters_approval:
        print(f"The tournament conditions are:")
        print(f"- population: {population}")
        print(f"- batch size: {state.batch_size}")
        print(f"- selection per batch: {state.batch_size}")
        print('-'*30)
        candidate_simulation(
            population = population,
            batch_size = state.batch_size,
            selected_per_batch = state.selected_per_batch
        )
        parameters_feedback = input("Do you approve this parameters?")
        if parameters_feedback.lower() in ['yes', 'y']:
            suggested_rounds = input("how many rounds do you wish to run?")
            state.number_of_rounds = int(suggested_rounds)
            parameters_approval = True
        else:
            print("Modify the parameters of the tournament")
            suggested_batch_size = input("What batch size do you wish to use?")
            suggested_selection_per_batch = input("How many candidates do you whish to select per batch?")
            state.batch_size = int(suggested_batch_size)
            state.selected_per_batch = int(suggested_selection_per_batch)
    return state
        
# 7. candidates tournament (improved version of the function score candidates)
@execution_time
@print_execution_status
def candidates_tournament(state : WorkflowState) -> WorkflowState:
    """selects the best candidates by comparing random groups of candidates"""
    # dictionary with the candidates 'id', 'resume' pairs
    candidates_to_evaluate = state.candidates_dict.copy()
    llm_selector = llm.with_structured_output(schema = GroupWinners)
    selection_chain = selection_prompt | llm_selector
    # full text of the criteria to evaluate candidates
    all_criteria = "\n- ".join(
        state.job_criteria.technical_skills +
        state.job_criteria.domains +  
        state.job_criteria.soft_skills + 
        state.job_criteria.culture
    )
    initial_groups = defaultdict(list)
    round_summaries = defaultdict(dict)
    round_rationale = defaultdict(dict)
    for round in range(state.number_of_rounds):
        candidates_ids = candidates_to_evaluate.keys()
        random.shuffle(candidates_ids)
        # the batches are full of ids of the candidates
        batches = [
            candidates_ids[i : state.batch_size + i] for i in range(0, len(candidates_to_evaluate) + 1, state.batch_size)
        ]
        winner_of_round = defaultdict(dict)
        for batch_index, batch in tqdm(enumerate(batches)):
            if round == 0:
                # store metadata about the initial group
                initial_groups[batch_index] = batch
            # string for the cvs in the groupx
            group_resumes_string = '\n---\n'.join([
                f"Candidate ID: {cid}\nName: {state.candidates_names.get(cid, 'Unknown')}\nResume: {state.candidates_dict[cid]}" for cid in batch
            ])
            # select only 1 candidate for batches with less candidates
            k = min(state.selected_per_batch, max(1, len(batch) - 1))
            try:
                selection_obj = selection_chain.invoke(input = {
                    "number_of_candidates_to_select" : k,
                    "all_candidate_ids" : ", ".join(batch),
                    "combined_criteria_list" : all_criteria,
                    "group_resumes" : group_resumes_string
                })
                summary = getattr(selection_obj, "overall_summary", "")
                # storing the rationale of the batch selection
                round_summaries[round][batch] = summary 
                for winner_data in selection_obj.selected_winners:
                    # storing the results
                    winner_id = winner_data.candidate_id
                    # state.round_rationale.setdefault(round, {})
                    round_rationale[round][winner_id] = winner_data.justification
                    # append the winner of round
                    winner_of_round[winner_id] = candidates_to_evaluate[winner_id]
            except Exception as e:
                print(f"Error evaluating the batch: {batch_index + 1} : {e}")
                # fallback alternative
                if batch:
                    winner_of_round[batch[0]] = candidates_to_evaluate[batch[0]]
        candidates_to_evaluate = winner_of_round
        print(f"Round finished. {len(candidates_to_evaluate)} candidates were promoted")
         
    # store the results from initial groups
    initial_groups_lookup = {
        cid : k for k, v in initial_groups.items() for cid in v # dictionary with ids : initial group values
    }
    state.round_rationale = round_rationale
    state.round_summaries = round_summaries
    state.initial_groups = initial_groups_lookup
    state.top_candidates = candidates_to_evaluate
    # export the rationales
    dump(
        obj = state.round_rationale,
        fp = open(os.path.join(state.scores_folder, "tournament_rationales.json"), mode = "w"),
        indent = 4
    )
    dump(
        obj = state.round_summaries,
        fp = open(os.path.join(state.scores_folder, "round_summaries.json"), mode = "w"),
        indent = 4
    )
    
    return state   
    
    # initial_groups = defaultdict[Any, dict](dict)
    # round_number = 0
    # # state.round_summaries = {}
    # # state.round_rationale = {}
    # round_summaries = defaultdict[Any, dict](dict)
    # round_rationale = defaultdict[Any, dict](dict)
    # # the loop ends with 10% of the candidates
    # limit = ceil(len(state.candidates_dict.keys()) * 0.1)
    # while len(candidates_to_evaluate) > limit + 1: 
    #     candidate_ids = list[str](candidates_to_evaluate.keys())
    #     random.shuffle(candidate_ids)
    #     winner_of_round = {}
    #     groups = [
    #         candidate_ids[i : i + state.batch_size] for i in range(0, len(candidate_ids) + 1, state.batches)
    #     ]
    #     for group_index, group_ids in tqdm[tuple[int, list[str]]](enumerate(groups)):
    #         # store the initial groups of the candidates
    #         if round_number == 0:
    #             initial_groups[group_index] = group_ids
    #         # string for the cvs in the group
    #         group_resumes_string = '\n---\n'.join([
    #             f"Candidate ID: {cid}\nName: {state.candidates_names.get(cid, 'Unknown')}\nResume: {state.candidates_dict[cid]}" for cid in group_ids
    #         ])
    #         try:
    #             selection_obj = selection_chain.invoke({
    #                 "all_candidate_ids" : ", ".join(group_ids),
    #                 "combined_criteria_list" : all_criteria,
    #                 "group_resumes" : group_resumes_string
    #             })
    #             # this object has two attributes: selected_winners and overall_summary
    #             # overall_summary is the overall explanation of the winners in the gruop
    #             # selected_winners is a list of candidate_id, justification
    #             # state.round_summaries.setdefault(round_number, {})
    #             summary = getattr(selection_obj, "overall_summary", "")
    #             round_summaries[round_number][group_index] = summary 
    #             # state.round_summaries[round_number][group_index] = selection_obj.overall_summary
    #             for winner_data in selection_obj.selected_winners:
    #                 # storing the results
    #                 winner_id = winner_data.candidate_id
    #                 # state.round_rationale.setdefault(round_number, {})
    #                 round_rationale[round_number][winner_id] = winner_data.justification
    #                 # append the winner of round
    #                 winner_of_round[winner_id] = candidates_to_evaluate[winner_id]
                    
    #         except Exception as e:
    #             print(f"Error evaluating group {group_index + 1}: {e}")
    #             # fall-back: if there is an error on the evaluation then the first candidate is selected
    #             if group_ids:
    #                 winner_of_round[group_ids[0]] = candidates_to_evaluate[group_ids[0]]
    #     round_number += 1
    #     candidates_to_evaluate = winner_of_round
    #     print(f"Round finished. {len(candidates_to_evaluate)} candidates were promoted")
    # # store the results from initial groups
    # initial_groups_lookup = {
    #     cid : k for k, v in initial_groups.items() for cid in v # dictionary with ids : initial group values
    # }
    # state.round_rationale = round_rationale
    # state.round_summaries = round_summaries
    # state.initial_groups = initial_groups_lookup
    # state.top_candidates = candidates_to_evaluate
    # # export the rationales
    # dump(
    #     obj = state.round_rationale,
    #     fp = open(os.path.join(state.scores_folder, "tournament_rationales.json"), mode = "w"),
    #     indent = 4
    # )
    # dump(
    #     obj = state.round_summaries,
    #     fp = open(os.path.join(state.scores_folder, "round_summaries.json"), mode = "w"),
    #     indent = 4
    # )
    
    # return state
    
# 6. ranking candidates
@execution_time
@print_execution_status
def score_candidates(state : WorkflowState) -> WorkflowState:
    """evaluate the top candidates from the tournament on the provided criteria"""
    llm_for_scores = llm.with_structured_output(schema = EvaluationScores)
    scoring_chain = scoring_prompt | llm_for_scores
    
    # Concatenating the criteria into a single list
    criteria = state.job_criteria
    all_criteria = (
        criteria.domains +
        criteria.technical_skills + 
        criteria.soft_skills + 
        criteria.culture
    )

    # Initialize a temporary dictionary to hold all scores
    all_candidates_scores = {}
    
    # Scoring the candidates
    for id, resume in state.top_candidates.items(): # evaluate only the top candidates instead of the complete pool
        try:
            # Ask the llm to evaluate the candidate based on their resume
            scores_object = scoring_chain.invoke({
                "job_description" : state.job_description,
                "candidate_resume" : resume,
                "all_criteria" : ", ".join(all_criteria)
            })
            # Get the scores from the response
            scores_list = scores_object.scores
        except Exception as e:
            print(f"Error generating the scores for candidate {id}: {e}")
            scores_list = ["N/A"] * len(all_criteria)
            
        # Generate the dictionary of "criteria" : score pairs
        candidate_scores_dict = {
            criterion: score for criterion, score in zip(all_criteria, scores_list)
        }
        
        # Update the temporary dictionary with the current candidate's scores
        all_candidates_scores[id] = candidate_scores_dict
    
    # Finally, update the state with the complete dictionary of all candidates' scores
    state.candidates_scores = all_candidates_scores

    return state

# 7. exporting candidates scores to csv, adding the batches
@print_execution_status
def export_scores(state : WorkflowState) -> WorkflowState:
    """Exports candidate results with their corresponding batches"""
    scores_df = scores_to_dataframe(
        candidates_scores = state.candidates_scores
    )
    # add the names colum
    scores_df_reset = scores_df.reset_index()
    scores_df_reset['names'] = scores_df_reset['candidate_id'].map(state.candidates_names)
    scores_df_reset['group_id'] = scores_df_reset['candidate_id'].map(state.initial_groups)
    scores_df = scores_df_reset.set_index(['candidate_id'])
    # export to csv
    scores_df.to_csv(
        path_or_buf = os.path.join(state.scores_folder, "scores.csv"),
        index = True
    )
    return state

# 8. calculating the overall score
@print_execution_status
def calculate_overall_score(state : WorkflowState) -> WorkflowState:
    """calculate the overall score of the candidates"""
    scores_df = read_csv(
        filepath_or_buffer = os.path.join(state.scores_folder, "scores.csv"),
        index_col = ["candidate_id", "group_id", "names"]
    )
    # print("Domains:", state.job_criteria.domains)
    # print("Technical skills:", state.job_criteria.technical_skills)
    # print("Soft skills:", state.job_criteria.soft_skills)
    # print("Culture:", state.job_criteria.culture)
    # print("CSV columns:", scores_df.columns.tolist())
    weights = state.job_criteria.weights
    # create the auxiliary columns for domain skills
    scores_df = auxiliary_col_calutation(
        df = scores_df,
        columns = state.job_criteria.domains,
        weight = weights[0]/100,
        output_col_name = "domains"
    )
    # create the columns for tech skills
    scores_df = auxiliary_col_calutation(
        df = scores_df,
        columns = state.job_criteria.technical_skills,
        weight = weights[1]/100,
        output_col_name = "tech_skills"
    )
    # create the auxiliary columns for soft skills
    scores_df = auxiliary_col_calutation(
        df = scores_df,
        columns = state.job_criteria.soft_skills,
        weight = weights[2]/100,
        output_col_name = "soft_skills"
    )
    # create the auxiliary columns for culture
    scores_df = auxiliary_col_calutation(
        df = scores_df,
        columns = state.job_criteria.culture,
        weight = weights[3]/100,
        output_col_name = "culture"
    )
    # calculate the refined average considering that some of the criteria can be null
    cols_2_sum = [c for c in scores_df.columns if c.endswith("average")]
    # calculate the refined average potential outliers
    scores_df = refined_overall_calculation(
        df = scores_df,
        columns = cols_2_sum,
        output_col_name = "overall"
    )
    # export the scores
    scores_df.to_csv(
        path_or_buf = os.path.join(state.scores_folder, "scores.csv"),
        index = True
    )
    return state

# 9. extracting candidate experience
@execution_time
@print_execution_status
def extract_experience(state: WorkflowState) -> WorkflowState:
    """extract the candidate experience and export to JSON"""
    llm_extractor = llm.with_structured_output(schema = CandidateExperience)
    extraction_chain = exp_extraction_prompt | llm_extractor
    
    # Initialize a temporary dictionary to hold all scores
    all_candidates_exp = {}
    
    # Scoring the candidates
    for id, resume in state.top_candidates.items():
        try:
            # Ask the llm to extract the candidate experience
            experience_object = extraction_chain.invoke({
                "resume" : resume
            }).model_dump() # dump the structured output into a dictionary
        except Exception as e:
            print(f"Error extracting the experience for the candidate '{id}': {e}")
            experience_object = {}
        # Update the temporary dictionary with the current candidate's experience
        all_candidates_exp[id] = experience_object
    
    # Update the state with the complete dictionary of all candidates' scores
    state.candidates_exp = all_candidates_exp
    
    # Export to a JSON file
    file_path = os.path.join(state.scores_folder, "candidates_experience.json")
    with open(file_path, mode = "w") as json_file:
        dump(
            obj = state.candidates_exp,
            fp = json_file,
            indent = 4
        )

    return state

# 10. creating the intreview questions for the best candidates
@print_execution_status
def generate_questions(state : WorkflowState) -> WorkflowState:
    """generate interview questions for the top candidates based solely on their experience"""
    llm_constrained = llm.with_structured_output(schema = InterviewQuestions)
    questions_chain = interview_questions_prompt | llm_constrained
    for cid, resume in state.top_candidates.items():
        msg = questions_chain.invoke({
            "job_description" : state.job_description,
            "candidate_experience" : format_experience_for_prompt(state.candidates_exp[cid])
        })
        tmp_dict = msg.model_dump()
        tmp_dict['name'] = state.candidates_names[cid]
        tmp_file_name = os.path.join(
            state.scores_folder,
            f"interview_questions_{state.candidates_names[cid]}.json"
        )
        dump(
            obj = tmp_dict,
            fp = open(tmp_file_name, mode = "w"),
            indent = 4
        )
    return state
        
# a. evaluation routing
@print_execution_status
def route_evalution(state : WorkflowState) -> Literal["accepted", "rejected"]:
    """Routes the evaluation based on the feedback and the counter"""
    # pruning some of the messages
    state.messages = state.messages[-4:]
    if state.criteria_status == "good" or state.count > 2:
        return "accepted"
    else:
        return "rejected"