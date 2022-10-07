from dataset import CustomDataset
import transformers
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import onnxruntime
import pandas as pd
import streamlit as st

@st.experimental_memo 
def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only = True)
    return tokenizer

def encode_text(text):
    tokenizer = get_tokenizer()
    data = pd.DataFrame(columns=['text'],data = [text])
    dataset = CustomDataset(data,tokenizer)
    dataloader = DataLoader(dataset)

    for data in dataloader:
        ids = data['ids'].cpu().detach().numpy()
        mask = data['mask'].cpu().detach().numpy()
        return ids,mask

def get_inputs(session,text):
    ids,mask = encode_text(text)
    return {
        session.get_inputs()[0].name: ids,
        session.get_inputs()[1].name: mask
    }

def get_prediction(session,input_dict):
    output = session.run(None,input_dict)
    prediction = output[0]
    labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    pred = []
    percent = []
    for p_val in prediction[0]:
        percent.append(f'{p_val*100:.1f}')
        if p_val>=0.5:
            pred.append(1)
        else:
            pred.append(0)

    result = {}
    for k,v in zip(labels,pred):
        result[k] = v

    return result,percent


if __name__ == '__main__':
    st.title('Toxic Comment Analyser')
    text = st.text_area(label='Comment',value='',max_chars=256,placeholder='Type or Paste Comment...')
    click = st.button(label='Analyse Comment')

    if click == True and len(text) != 0:
        session = onnxruntime.InferenceSession("onnx-model/toxic_comment.onnx")
        session_input = get_inputs(session, text)
        result,percent = get_prediction(session, session_input)
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        if result['toxic'] == 1:
            col1.metric('Toxic',result['toxic'],f'{percent[0]}%',delta_color='inverse')
        else:
            col1.metric('Toxic',result['toxic'],f'{percent[0]}%',delta_color='normal')
        
        if result['severe_toxic'] == 1:
            col2.metric('Severely Toxic',result['severe_toxic'],f'{percent[1]}%',delta_color='inverse')
        else:
            col2.metric('Severely Toxic',result['severe_toxic'],f'{percent[1]}%',delta_color='normal')
        
        if result['obscene'] == 1:
            col3.metric('Obscene',result['obscene'],f'{percent[2]}%',delta_color='inverse')
        else:
            col3.metric('Obscene',result['obscene'],f'{percent[2]}%',delta_color='normal')
        
        if result['threat'] == 1:
            col4.metric('Threat',result['threat'],f'{percent[3]}%',delta_color='inverse')
        else:
            col4.metric('Threat',result['threat'],f'{percent[3]}%',delta_color='normal')

        if result['insult'] == 1:
            col5.metric('Insult',result['insult'],f'{percent[4]}%',delta_color='inverse')
        else:
            col5.metric('Insult',result['insult'],f'{percent[4]}%',delta_color='normal')

        if result['identity_hate'] == 1:
            col6.metric('Identity Hate',result['identity_hate'],f'{percent[5]}%',delta_color='inverse')
        else:
            col6.metric('Identity Hate',result['identity_hate'],f'{percent[5]}%',delta_color='normal')

    elif click==True and len(text)==0:
        st.error('Comment not found. Please type comment.',icon='🚨')


