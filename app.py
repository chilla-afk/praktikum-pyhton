import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ==================================================
# DATASET
# ==================================================

iris = datasets.load_iris()

X = iris.data
y = iris.target
species = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================================
# MODELS
# ==================================================

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

trained = {}
scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    trained[name] = model
    scores[name] = round(acc * 100, 2)

# ==================================================
# CHART
# ==================================================

def make_chart():
    df = pd.DataFrame({
        "Model": list(scores.keys()),
        "Accuracy": list(scores.values())
    })

    fig = px.bar(
        df,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        text="Accuracy",
        template="plotly_dark",
        height=400
    )

    return fig

# ==================================================
# PREDICTION
# ==================================================

def predict(model_name, sl, sw, pl, pw):
    model = trained[model_name]

    data = np.array([[sl, sw, pl, pw]])
    pred = model.predict(data)[0]

    flower = species[pred]
    score = scores[model_name]

    return (
        f"🌸 Predicted: {flower.upper()}",
        f"🎯 Accuracy Model: {score}%"
    )

# ==================================================
# BEST MODEL
# ==================================================

best_model = max(scores, key=scores.get)

# ==================================================
# UI
# ==================================================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🌸 IRIS INTELLIGENCE LAB
    ### Smart Flower Prediction Dashboard
    """)

    with gr.Row():
        gr.Markdown(f"""
        ### 📌 Total Dataset: 150  
        ### 🤖 Models: 3  
        ### 🏆 Best Model: {best_model}  
        """)

    gr.Markdown("---")

    with gr.Row():

        with gr.Column():

            model = gr.Dropdown(
                choices=list(models.keys()),
                value="Decision Tree",
                label="Choose Model"
            )

            sl = gr.Slider(4,8,value=5.1,label="Sepal Length")
            sw = gr.Slider(2,5,value=3.5,label="Sepal Width")
            pl = gr.Slider(1,7,value=1.4,label="Petal Length")
            pw = gr.Slider(0.1,3,value=0.2,label="Petal Width")

            btn = gr.Button("🚀 Predict")

        with gr.Column():

            out1 = gr.Textbox(label="Prediction")
            out2 = gr.Textbox(label="Performance")

    btn.click(
        predict,
        inputs=[model, sl, sw, pl, pw],
        outputs=[out1, out2]
    )

    gr.Markdown("## 📊 Model Analytics")

    gr.Plot(value=make_chart())

app.launch()