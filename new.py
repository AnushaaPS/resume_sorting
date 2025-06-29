import os
import re
import csv
import torch
import numpy as np
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to calculate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to calculate cosine similarity score
def calculate_similarity_score(text1, text2):
    embedding1 = get_bert_embedding(text1)
    embedding2 = get_bert_embedding(text2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

# Function to parse text from PDF using PdfReader
def parse_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text

# Function to parse text from DOCX
def parse_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to parse text from TXT
def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to parse resume and extract relevant details
def parse_resume(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        resume_text = parse_pdf(file_path)
    elif file_extension.lower() == '.docx':
        resume_text = parse_docx(file_path)
    elif file_extension.lower() == '.txt':
        resume_text = parse_txt(file_path)
    else:
        print(f"Unsupported file format: {file_extension}. Skipping {file_path}.")
        return None
    return resume_text

# Function to extract skills from the resume text
def extract_skills(resume_text, job_skills):
    resume_skills = []
    for skill in job_skills:
        if re.search(re.escape(skill), resume_text, re.IGNORECASE):
            resume_skills.append(skill)
    return resume_skills

# Function to extract education details from the resume text
def extract_education(resume_text, job_title):
    education_details = []
    education_patterns = {
        'software engineer': [
            r'(Bachelor(?:\'s)?\s(?:of\s)?(?:Computer Science|Engineering|Technology|Science|B\.Sc|B\.Eng|B\.Tech|B\.E))',
            r'(Master(?:\'s)?\s(?:of\s)?(?:Computer Science|Engineering|Technology|Science|M\.Sc|M\.Eng|M\.Tech|M\.E))',
            r'(PhD|Doctor(?:ate)?\s(?:of\s)?(?:Computer Science|Engineering|Technology|Science|D\.Sc|D\.Eng|D\.Tech))'
        ],
        # Define more job titles and their relevant education patterns as needed
    }

    if job_title.lower() in education_patterns:
        patterns = education_patterns[job_title.lower()]
        for pattern in patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                education_details.append(match.strip())

    return ', '.join(education_details)

# Function to extract experience details from the resume text
def extract_experience(resume_text):
    total_months_experience = 0.0

    # Regular expression pattern to match date ranges like 'Feb 2023-Current'
    date_range_pattern = r"([A-Za-z]{3}\s*\d{4})\s*-\s*(Current|[A-Za-z]{3}\s*\d{4})"

    # Find all date ranges in the resume text
    date_ranges = re.findall(date_range_pattern, resume_text)
    if date_ranges:
        for start_date, end_date in date_ranges:
            start_month, start_year = parse_date(start_date)
            end_month, end_year = parse_date(end_date)

            if start_month and start_year:
                start_date = datetime(start_year, start_month, 1)
                if end_date.lower().strip() == "current":
                    end_date = datetime.now()
                else:
                    end_date = datetime(end_year, end_month, 1)
                duration = (end_date - start_date).days / 30.4
                total_months_experience += duration

    total_years_experience = total_months_experience / 12

    return total_years_experience

# Function to parse a date in format like 'Feb 2023' and return month and year
def parse_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%b %Y")
        return date_obj.month, date_obj.year
    except ValueError:
        try:
            date_obj = datetime.strptime(date_str, "%B %Y")
            return date_obj.month, date_obj.year
        except ValueError:
            print(f"Error parsing date: {date_str}")
            return None, None

# Function to process resumes and calculate scores
def process_resumes(resumes_folder, job_description):
    candidates_scores = []

    job_qualifications = job_description.get('qualifications', [])
    job_skills = job_description.get('skills', [])
    job_title = job_description.get('job_title', '').lower()
    required_experience = job_description.get('experience_years', 0)

    # Iterate through each file in the folder
    for resume_file in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, resume_file)
        if os.path.isfile(file_path):
            resume_text = parse_resume(file_path)
            if resume_text:
                # Extract details from the resume
                candidate_name = os.path.splitext(resume_file)[0]
                resume_skills = extract_skills(resume_text, job_skills)
                education_details = extract_education(resume_text, job_title)
                total_years_experience = extract_experience(resume_text)

                # Calculate heuristic scores
                skills_score = len(resume_skills) / len(job_skills) * 5
                education_score = calculate_education_score(education_details, job_title)
                experience_score = min(total_years_experience / required_experience * 10, 10) if required_experience > 0 else min(total_years_experience / 5 * 10, 10)

                # Calculate total score (including skill, education, and experience scores)
                total_score = skills_score + education_score + experience_score

                # Append candidate's scores to list
                candidates_scores.append({
                    'CandidateName': candidate_name,
                    'SkillsScore': skills_score,
                    'EducationScore': education_score,
                    'ExperienceScore': experience_score,
                    'TotalScore': total_score
                })

    # Sort candidates by TotalScore in descending order
    candidates_scores_sorted = sorted(candidates_scores, key=lambda x: x['TotalScore'], reverse=True)

    # Print the top suitable candidates
    print("Top Suitable Candidates:")
    for idx, candidate in enumerate(candidates_scores_sorted, start=1):
        print(f"Rank {idx}: {candidate['CandidateName']} - Total Score: {candidate['TotalScore']}")

    # Save sorted scores to CSV
    with open('resume_scores_sorted.csv', 'w', newline='') as csvfile:
        fieldnames = ['CandidateName', 'SkillsScore', 'EducationScore', 'ExperienceScore', 'TotalScore']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates_scores_sorted:
            writer.writerow(candidate)

    print("Resume scoring completed. Results saved in 'resume_scores_sorted.csv'.")

def calculate_education_score(education_details, job_title):
    education_score = 0.0

    # Define relevant qualifications based on job title
    relevant_qualifications = {
        'software engineer': ['computer science', 'engineering', 'technology']
        # Add more job titles and relevant qualifications as needed
    }

    # Normalize and split education details into individual qualifications
    resume_qualifications = [qual.strip().lower() for qual in education_details.split(',')]

    # Check each resume qualification against suitable qualifications
    for resume_qual in resume_qualifications:
        for relevant_qual in relevant_qualifications.get(job_title, []):
            if relevant_qual in resume_qual or resume_qual in relevant_qual:
                education_score = 5.0
                return education_score

    return education_score

# Main function to get user input for job description and resumes folder
def main():
    # Get job description from user
    print("Enter job description details:")
    job_title = input("Enter the job title: ").strip()
    qualifications = input("Enter necessary qualifications (separated by comma): ").strip().split(',')
    skills = input("Enter required skills (separated by comma): ").strip().split(',')
    experience_text = input("Enter required years of experience: ").strip()

    # Extract numeric experience years from the input text
    experience_years_match = re.search(r'(\d+)\s*\-?\s*(\d+)?\s*years?', experience_text)
    if experience_years_match:
        experience_years = (int(experience_years_match.group(1)) + int(experience_years_match.group(2) or 0)) / 2
    else:
        experience_years = 0

    job_description = {
        'job_title': job_title,
        'qualifications': [qual.strip() for qual in qualifications],
        'skills': [skill.strip() for skill in skills],
        'experience_years': experience_years
    }

    # Get resumes folder from user
    resumes_folder = input("Enter path to folder containing resumes: ").strip()

    # Process resumes
    process_resumes(resumes_folder, job_description)

if __name__ == "__main__":
    main()
