import streamlit as st
import plotly.graph_objects as go

import dataprep
import evals
import plots

from optim import train_model
from multilayer_net import Net

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("./mystyle.css")

st.title("Training a classifier")
st.markdown("---")
st.markdown("### Specify the training setup")
col1, col2, col3 = st.columns(3)
with col1:
    inp_epochs = st.number_input(label='epochs', min_value=100, max_value=10000, value=200, step=100)
with col2:
    inp_optim = st.selectbox(label="optimizer", options=['Adam', 'SGD'])
with col3:
    inp_lr = st.selectbox(label="learning rate", options=[1e-1, 1e-2, 1e-3, 1e-4])

st.markdown("---")
st.markdown("### Specify the network configuration here")
col4, col5, col6 = st.columns(3)
with col4:
    inp_layer1 = st.selectbox(label="Layer 1", key="select1", options=[None]+[i for i in range(10)], index=3)
with col5:
    inp_layer2 = st.selectbox(label="Layer 2", key="select2", options=[None]+[i for i in range(10)])
with col6:
    inp_layer3 = st.selectbox(label="Layer 3", key="select3", options=[None]+[i for i in range(10)])
st.text("This is a test")

st.markdown('---')
inp_do_train = st.button("Click here to train now")
my_bar = st.progress(0)


# tab1, tab2 = st.tabs(["loss curves", "cm matrix"])
# with tab1:
#     info_train = st.empty()
#     plot_train = st.plotly_chart(go.Figure())
# with tab2:
#     plot_cm = st.plotly_chart(go.Figure())

if inp_do_train:

    layers=[inp_layer1, inp_layer2, inp_layer3]
    layers=[i for i in layers if i is not None]

    model = Net(layers)
    data = dataprep.read_data()

    with st.spinner("Running optimization"):
        net, epoch_data = train_model(
            model = model,
            data = data,
            lr=inp_lr,
            optimizer=inp_optim,
            num_epochs=inp_epochs
        )


        fig = plots.plot_train_curves(epoch_data)
        st.plotly_chart(fig)


        evaluator = evals.Evaluator()
        cm = evaluator.confusion_matrix(net)
        fig = plots.plot_confusion_matrix(cm, ["a", "b"], "This is a title")
        st.plotly_chart(fig)
