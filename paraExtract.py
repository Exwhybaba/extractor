import pandas as pd
import streamlit as st
import nltk
from pdfReader import extract

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to extract entities and save as CSV
def entity(text, filename):
    current_question = 1
    question = []
    alpha = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.', 'K.', 
             'L.', 'M.', 'N.', 'O.', 'P.', 'Q.', 'R.', 'S.', 'T.', 'U.', 'V.', 'W.', 'X.', 'Y.', 'Z.',
            'a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.', 'k.', 'l.', 'm.', 'n.', 'o.', 
             'p.', 'q.', 'r.', 's.', 't.', 'u.', 'v.', 'w.', 'x.', 'y.', 'z.']
    for line in text:
        line = line.strip()  # Remove leading and trailing whitespaces
        if line.startswith(str(current_question) + '.'):
            # Extract the question and add it to the parameters dictionary
            question.append(f'Q{current_question}')
        elif line.strip().startswith(tuple(alpha)):
            question.append(f'Q{current_question}{line[0]}')
        elif line.startswith(str(current_question + 1) + '.'):
            current_question += 1
            question.append(f'Q{current_question}')

    # Tokenize the text
    tokens = nltk.word_tokenize(" ".join(text))
    # Part-of-speech tagging
    tagged = nltk.pos_tag(tokens)
    # Named entity recognition
    named_entities = nltk.ne_chunk(tagged)

    # Initialize a dictionary to store extracted entities
    extracted_entities = {"STUDY": [], "FORM": [], "CENTER": [], "PATIENT": [], 
                          "NAMECODE": [], "FORMCODE": [], "DATECOMP": [], "WEEK": []}

    # Extract named entities from the text
    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree):
            if entity.label() in extracted_entities:
                extracted_entities[entity.label()].append(' '.join([word for word, _ in entity.leaves()]))

    # Create a list of strings representing extracted entities without the colon
    entity_list = list(extracted_entities.keys())

    # Extend entity_list with questions
    entity_list.extend(question)

    # Create a DataFrame with columns as entity_list
    df = pd.DataFrame(columns=entity_list)

    # Add data to the DataFrame (here, all cells are filled with 'NA')
    df.loc[0] = ['NA'] * len(entity_list)

    # Save DataFrame to CSV with filename based on the uploaded file name
    df.to_csv(f'extract_{filename}.csv', index=False)

# Function to create a visually appealing user interface
def main():
    st.set_page_config(page_title="PDF Text Extractor", layout="wide")
    st.title("PDF Text Extractor")

    # Allow user to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], help="Please upload a PDF file.")

    if uploaded_file is not None:
        # Process the uploaded file and extract text
        text = extract(uploaded_file)
        
        # Extract filename
        filename = uploaded_file.name.split('.')[0]  # Extract filename without extension
        
        # Display the extracted text
        st.header("Extracted Text:")
        for line in text:
            st.write(line)

        # Add a download button for the extracted data
        if st.button("Download Extracted Data", help="Click here to download the extracted data as a CSV file."):
            # Extract and save the data
            entity(text, filename)
            # Provide download link
            with open(f'extract_{filename}.csv', 'rb') as f:
                st.download_button(label=f'Download {filename}.csv', data=f, file_name=f'extract_{filename}.csv', mime='text/csv')

if __name__ == "__main__":
    main()
