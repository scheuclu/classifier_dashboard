from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_train_curves(epoch_data):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    trace_train = go.Scatter(x=epoch_data["epochs"], y=epoch_data["train"], name='train')
    trace_test =  go.Scatter(x=epoch_data["epochs"], y=epoch_data["test"], name='test')
    trace_precision =  go.Scatter(x=epoch_data["epochs"], y=epoch_data["precision"], name='precision')
    trace_recall =  go.Scatter(x=epoch_data["epochs"], y=epoch_data["recall"], name='recall')
    fig.add_trace(trace_train, secondary_y=False)
    fig.add_trace(trace_test, secondary_y=False)
    fig.add_trace(trace_precision, secondary_y=True)
    fig.add_trace(trace_recall, secondary_y=True)
    fig.update_layout(
        go.Layout(yaxis=dict(type='log'))
    )
    return fig


def plot_confusion_matrix(cm, labels, title):
    data = go.Heatmap(z=cm, y=labels, x=labels, colorscale='gray')
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "white"},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    return fig