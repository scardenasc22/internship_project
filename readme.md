# Candidate Evaluation Workflow

An automated candidate evaluation system built with LangGraph that processes job descriptions and candidate resumes to generate evaluation criteria, scores, interview questions, and candidate feedback.

## Overview

This workflow automates the candidate screening and evaluation process by:

1. **Extracting company information** from job descriptions
2. **Generating evaluation criteria** (domains, technical skills, soft skills, culture) with weighted importance
3. **Running a tournament-style candidate selection** process to filter candidates
4. **Scoring candidates** based on the evaluation criteria
5. **Generating interview questions** tailored to each candidate and role level
6. **Providing candidate overviews** with strengths and weaknesses analysis

## Prerequisites

- Python 3.10.4 or higher
- Poetry (for dependency management) or pip
- OpenAI API key (for LLM access)
- Tavily API key (optional, for web search functionality)

## Installation

### Using Poetry (Recommended)

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Using pip

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Environment Variables

Create a `.env` file in the root directory of the project with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # Optional
```

The `.env` file is automatically loaded by the application using `python-dotenv`.

## Project Structure

```
internship_project/
├── src/
│   ├── graph.py             # Main workflow graph definition and execution
│   ├── nodes.py              # Workflow node implementations
│   ├── schemas.py            # Pydantic models for data validation
│   ├── prompts.py            # LLM prompts for various tasks
│   └── functions.py          # Utility functions
├── notebooks/                # Jupyter notebooks for experimentation
├── pyproject.toml            # Poetry configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Running the Program

### Basic Usage

Run the workflow with default parameters:

```bash
python3 src/graph.py
```

### Command-Line Arguments

The workflow supports various command-line arguments for customization:

```bash
python3 src/graph.py [OPTIONS]
```

#### Available Options

- `--job-description PATH`: Path to the job description text file
  - Default: `data/raw/job/data_role_des.txt`
  
- `--cv-folder PATH`: Path to the folder containing candidate resumes
  - Default: `data/raw/cv`
  
- `--scores-folder PATH`: Path to the output folder for results
  - Default: `data/gpt_5_results`
  
- `--batch-size N`: Number of candidates per tournament batch
  - Default: `5`
  
- `--selected-per-batch N`: Number of candidates to select as winners per batch
  - Default: `2`

#### Examples

```bash
# Use all defaults
python3 src/graph.py

# Custom candidates folder
python3 src/graph.py --cv-folder path/to/resumes/folder

# Custom job description
python3 src/graph.py --job-description path/to/job_description.txt

# Multiple custom parameters
python3 src/graph.py \
    --cv-folder data/raw/cv \
    --job-description data/raw/job/data_role_des.txt \
    --batch-size 10 \
    --selected-per-batch 3 \
    --scores-folder data/custom_results

# Custom output folder
python3 src/graph.py --scores-folder data/custom_results
```

## Workflow Steps

The workflow executes the following steps in sequence:

1. **Company Info Extraction** (`get_company_info`)
   - Extracts company details (name, location, sector, description) from the job description

2. **Criteria Generation** (`criteria_generation`)
   - Generates evaluation criteria with weights:
     - Domains (subject areas/industries)
     - Technical skills (technologies, tools)
     - Soft skills (interpersonal abilities)
     - Culture (company values alignment)

3. **LLM Criteria Evaluation** (`llm_criteria_evaluation`)
   - Evaluates the generated criteria for quality and completeness

4. **Human Criteria Assessment** (`human_criteria_assessment`)
   - Prompts for human approval of the criteria
   - Allows refinement with feedback

5. **Candidate Info Loading** (`get_candidates_info`)
   - Loads all candidate resumes from the specified folder
   - Extracts candidate names and creates candidate dictionaries

6. **Tournament Simulation** (`tournament_simulation`)
   - Simulates and displays the tournament structure
   - Shows how many candidates will be selected in each round

7. **Candidates Tournament** (`candidates_tournament`)
   - Runs the tournament selection process
   - Groups candidates into batches and selects winners
   - Stores selection rationale for each round

8. **Score Candidates** (`score_candidates`)
   - Scores each candidate based on the evaluation criteria
   - Uses LLM to assign scores (0-100) for each criterion

9. **Export Scores** (`export_scores`)
   - Exports individual scores to CSV files

10. **Calculate Overall Score** (`calculate_overall_score`)
    - Calculates weighted overall scores for each candidate

11. **Extract Experience** (`extract_experience`)
    - Extracts structured experience from resumes
    - Creates job history with titles, companies, dates, and responsibilities

12. **Generate Questions** (`generate_questions`)
    - Generates interview questions tailored to each candidate:
      - 4 experience questions (role-level appropriate)
      - 4 situational questions
      - 4 technical questions (role-level appropriate)

13. **Generate Candidate Overview** (`generate_candidate_overview`)
    - Creates comprehensive candidate feedback:
      - Overview summary
      - 3 strengths
      - 3 weaknesses

## Interactive Prompts

During execution, you may be prompted to:

### 1. Approve Criteria
Review and approve the generated evaluation criteria:
- Type `yes` to approve and proceed
- Type `no` to reject and provide feedback for refinement

### 2. Tournament Parameters
Review tournament simulation output and specify parameters:
- Review the tournament structure (number of rounds, candidates per round)
- Type `yes` to approve the parameters
- Specify how many rounds to run

## Output Files

Results are saved in the `scores_folder` directory (default: `data/gpt_5_results/`):

- **`scores.csv`**: Individual scores for each candidate on each criterion
- **`overall_scores.csv`**: Overall weighted scores for each candidate
- **`candidates_experience.json`**: Structured experience data for all candidates
- **`interview_questions_*.json`**: Individual candidate interview questions files
- **`strengts_and_weaknesses_*.json`**: Individual candidate feedback files containing:
  - Overview
  - Strengths (3 items)
  - Weaknesses (3 items)
- **`tournament_rationales.json`**: Justification for tournament selections
- **`round_summaries.json`**: Summary of each tournament round

## Supported File Formats

- **Resumes**: PDF (`.pdf`) and DOCX (`.docx`)
- **Job Descriptions**: Plain text (`.txt`)

## Configuration Options

### Changing the LLM Model

The default LLM model is set in `src/nodes.py` (line 62). Available models include:

- `gpt-4o-mini-2024-07-18` (GPT_4O_MINI)
- `gpt-4o-2024-08-06` (GPT_4O)
- `o3-mini-2025-01-31` (GPT_O3_MINI)
- `gpt-5-mini-2025-08-07` (GPT_5_MINI)
- `gpt-5-2025-08-07` (GPT_5)

To change the model, modify the `llm` variable in `src/nodes.py`:

```python
llm = ChatOpenAI(
    model = GPT_4O  # Change to your preferred model
)
```

### Random Seed

The workflow uses a seeded random number generator (seed: 2000) for reproducibility. This can be modified in `src/nodes.py` (line 49).

## Dependencies

Key dependencies include:

- **`langchain` & `langgraph`**: Workflow orchestration and state management
- **`langchain-openai`**: OpenAI LLM integration
- **`pydantic`**: Data validation and schema definitions
- **`pdfplumber`**: PDF text extraction
- **`python-docx`**: DOCX text extraction
- **`pandas`**: Data processing and CSV export
- **`tqdm`**: Progress bars for batch processing
- **`langchain-tavily`**: Web search functionality (optional)

See `requirements.txt` or `pyproject.toml` for the complete list of dependencies.

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your `.env` file contains valid API keys
   - Check that the `.env` file is in the project root directory
   - Verify the API keys are correct and have sufficient credits

2. **File Not Found Errors**
   - Check that file paths in command-line arguments are correct
   - Ensure the job description file exists
   - Verify the candidates folder contains resume files

3. **Import Errors**
   - Make sure you're running from the project root directory
   - Verify that all dependencies are installed
   - Check that you're using the correct Python version (3.10.4+)

4. **Path Issues When Running from Different Directories**
   - Always run the script from the project root directory
   - Use absolute paths or paths relative to the project root
   - Or add the project root to your Python path:
     ```python
     import sys
     sys.path.append('/path/to/internship_project')
     ```

### Getting Help

Run the help command to see all available options:

```bash
python3 src/graph.py --help
```

## Notes

- The workflow maintains state throughout execution using LangGraph's state management
- Tournament selection uses a seeded random number generator for reproducibility
- All LLM outputs are validated using Pydantic schemas to ensure data quality
- The workflow is designed to be interactive, allowing human oversight at critical decision points

## License

This project is proprietary software. All rights reserved.

Copyright (c) 2025 Synogize

See the [LICENSE](LICENSE) file for full terms and conditions.

For licensing inquiries, please contact: scardenasc22@gmail.com

## Author

Santiago Cardenas (scardenasc22@gmail.com)

