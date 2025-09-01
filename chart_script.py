import plotly.graph_objects as go
import json

# Data for the workflow steps
workflow_data = {
    "workflow_steps": [
        {"step": 1, "name": "Video Upload", "description": "User uploads 2-way, 3-way, or 4-way junction video"},
        {"step": 2, "name": "Preprocessing", "description": "Frame extraction and resizing"},
        {"step": 3, "name": "Vehicle Detection", "description": "YOLOv8 detection"},
        {"step": 4, "name": "Vehicle Classification", "description": "Car, Bus, Truck, Bike, Auto"},
        {"step": 5, "name": "Emergency Vehicle Recognition", "description": "Ambulance, Police, Fire Truck"},
        {"step": 6, "name": "Traffic Analysis", "description": "Count per type, density, alerts"},
        {"step": 7, "name": "Visualization", "description": "Graphs, reports, UI"},
        {"step": 8, "name": "Output Results", "description": "JSON/CSV + on-screen display"}
    ]
}

# Brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325', '#944454']

# Create figure
fig = go.Figure()

# Position parameters
box_width = 2.2
box_height = 0.7
y_spacing = 1.2
start_y = 6.5

# Add boxes and text for each step
for i, step in enumerate(workflow_data["workflow_steps"]):
    y_pos = start_y - (i * y_spacing)
    color = colors[i % len(colors)]
    
    # Abbreviate names to fit 15 character limit
    name = step["name"]
    if name == "Vehicle Detection":
        name = "Vehicle Detect"
    elif name == "Vehicle Classification":
        name = "Vehicle Class"
    elif name == "Emergency Vehicle Recognition":
        name = "Emergency Rec"
    elif name == "Output Results":
        name = "Output Results"
    
    # Abbreviate description to fit character limits while being accurate
    desc = step["description"]
    if "Frame extraction" in desc:
        desc = "Extract & resize"
    elif "YOLOv8" in desc:
        desc = "YOLOv8 detect"
    elif "Car, Bus, Truck" in desc:
        desc = "Car/Bus/Truck+"
    elif "Ambulance, Police" in desc:
        desc = "Ambulance/Police"
    elif "Count per type" in desc:
        desc = "Count/density"
    elif "Graphs, reports" in desc:
        desc = "UI & graphs"
    elif "JSON/CSV" in desc:
        desc = "JSON/CSV out"
    elif "User uploads" in desc:
        desc = "Upload video"
    
    # Add rectangle shape for box
    fig.add_shape(
        type="rect",
        x0=-box_width/2, y0=y_pos-box_height/2,
        x1=box_width/2, y1=y_pos+box_height/2,
        fillcolor=color,
        line=dict(color="white", width=2),
        opacity=0.9
    )
    
    # Add step number and name
    fig.add_annotation(
        x=0, y=y_pos+0.12,
        text=f"<b>{step['step']}. {name}</b>",
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0)"
    )
    
    # Add description
    fig.add_annotation(
        x=0, y=y_pos-0.12,
        text=desc,
        showarrow=False,
        font=dict(size=10, color="white"),
        bgcolor="rgba(0,0,0,0)"
    )
    
    # Add arrow to next step (except for last step)
    if i < len(workflow_data["workflow_steps"]) - 1:
        arrow_start_y = y_pos - box_height/2
        arrow_end_y = y_pos - y_spacing + box_height/2
        
        # Add arrow line
        fig.add_shape(
            type="line",
            x0=0, y0=arrow_start_y,
            x1=0, y1=arrow_end_y,
            line=dict(color="#333333", width=3)
        )
        
        # Add arrowhead
        fig.add_annotation(
            x=0, y=arrow_end_y,
            ax=0, ay=arrow_start_y - 0.1,
            arrowhead=2,
            arrowsize=1.8,
            arrowwidth=4,
            arrowcolor="#333333",
            showarrow=True,
            text=""
        )

# Update layout
fig.update_layout(
    title="Traffic Analyzer Project Flow",
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-2.5, 2.5]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-2.5, 7.5]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Add a dummy trace (required by plotly)
fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(opacity=0), showlegend=False))

# Save the chart
fig.write_image("traffic_analyzer_workflow.png", width=600, height=800)