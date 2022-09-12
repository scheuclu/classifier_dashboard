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
        go.Layout(
            title="Training Curves",
            titlefont=dict(color="#11A27B"),
            yaxis=dict(
                title="loss values",
                type='log',
                linecolor='#11A27B',
                gridcolor='#11A27B',
                titlefont=dict(color="#11A27B"),
                tickfont=dict(color="#11A27B"),
                linewidth=2,
            ),
            yaxis2=dict(
                type='log',
                linecolor='#11A27B',
                gridcolor='#11A27B',
                titlefont=dict(color="#11A27B"),
                tickfont=dict(color="#11A27B"),
                linewidth=2
            ),
            xaxis=dict(
                title='epoch',
                linecolor='#11A27B',
                gridcolor='#11A27B',
                titlefont=dict(color="#11A27B"),
                tickfont=dict(color="#11A27B"),
                linewidth=2),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    return fig


def plot_confusion_matrix(cm, labels, title):
    colorscale=[
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, "#273346"],
        [1.0, "#4B9C29"]
    ]
    data = go.Heatmap(z=cm, y=labels, x=labels, colorscale=colorscale, showscale=False)
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

    layout = go.Layout(
        title="Confusion Matrix",
        titlefont=dict(color="#11A27B"),
        annotations=annotations,
        yaxis=dict(
            title="actual class",
            linecolor='#11A27B',
            gridcolor='#11A27B',
            titlefont=dict(color="#11A27B"),
            tickfont=dict(color="#11A27B"),
            linewidth=2,
        ),
        xaxis=dict(
            title='predicted class',
            linecolor='#11A27B',
            gridcolor='#11A27B',
            titlefont=dict(color="#11A27B"),
            tickfont=dict(color="#11A27B"),
            linewidth=2),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = go.Figure(data=data, layout=layout)

    return fig
