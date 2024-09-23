import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Load data
data_path = r'Overall_Averages.xlsx'
df = pd.read_excel(data_path)

# Define maximum scores for the columns (assuming similar columns to CODE 1)
max_scores = {
    "EST I total": 1600,
    "EST I - Literacy": 800,
    "EST I - Mathematics": 800,
    "EST I - Essay": 8,
    "EST II - Biology": 80,
    "EST II - Physics": 75,
    "EST II - Chemistry": 85,
    "EST II - Math 1": 50,
    "EST II - Math 2": 50,
    "EST II - Literature": 60,
    "EST II - World History": 65,
    "EST II - Economics": 60
}

# Initialize the app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Student Performance Dashboard", className='text-center mb-4'), width=12)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='student-search',
            options=[{'label': username, 'value': username} for username in df['Username'].unique()],
            placeholder='Search for a student by username',
            className='mb-4',
            multi=False,
            searchable=True
        ), width=6),

        dbc.Col(dcc.Dropdown(
            id='test-dropdown',
            options=[{'label': version, 'value': version} for version in df['Test'].unique()],
            multi=True,
            placeholder='Select Test(s)',
            className='mb-4'
        ), width=6),

        dbc.Col(dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': country, 'value': country} for country in df['Country'].unique()],
            multi=True,
            placeholder='Select country(ies)',
            className='mb-4'
        ), width=6),

        dbc.Col(dcc.Dropdown(
            id='test-version-dropdown',
            options=[{'label': 'Select All Versions', 'value': 'ALL'}] +
                    [{'label': version, 'value': version} for version in df['Version'].unique()],
            placeholder='Select test version(s)',
            className='mb-4',
            multi=True
        ), width=6)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(html.Div(id='gauges-container', className='d-flex flex-wrap justify-content-center'))
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id='totals-container', className='text-center mt-5'))
    ])
], fluid=True, style={'max-width': '1100px', 'margin': '0 auto'})  # Set max-width for container


# Function to wrap text at spaces
def wrap_text(text, max_length=35):
    # Remove unwanted prefixes before wrapping
    text = text.replace('A-SK-', '').replace('B-SK-', '').replace('C-SK-', '').replace('D-SK-', '')
    text = text.replace('A-', '').replace('B-', '').replace('C-', '').replace('D-', '')
    
    words = text.split()
    lines = []
    line = ""

    for word in words:
        if len(line) + len(word) + 1 <= max_length:
            if line:
                line += " "
            line += word
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    return '<br>'.join(lines)


# Function to create gauge sections by test
def create_gauge_sections(filtered_df):
    sections = []
    tests = filtered_df['Test'].unique()

    for test in tests:
        test_df = filtered_df[filtered_df['Test'] == test]
        gauges = []
        skills = test_df['Skill/Passage'].unique()

        skill_gauges = []
        non_skill_gauges = []

        for skill in skills:
            skill_row = test_df[test_df['Skill/Passage'] == skill].iloc[0]
            average_score = skill_row['Average Score']
            percentage_score = average_score * 100

            # Determine the color based on whether the score is the highest or lowest in its category
            bar_color = 'blue'
            number_color = bar_color  # Use the same color for the displayed value

            # Wrap long titles
            title = wrap_text(skill)

            gauge = dcc.Graph(
                id=f'gauge-{skill}',
                figure=go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=percentage_score,
                    number={'font': {'size': 20, 'color': number_color}},  # Adjust font size and color here
                    title={
                        'text': title,
                        'font': {'size': 12 if len(title.split('<br>')) > 1 else 14}  # Adjust font size based on title length
                    },
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': bar_color}
                    }
                )),
                style={'display': 'inline-block', 'width': '250px', 'height': '250px', 'margin': '10px'}
            )

            skill_gauges.append(dbc.Col(gauge, width=3))

        section = dbc.Row([
            dbc.Col(html.H3(test, className='text-center my-4'), width=12),
            dbc.Col(dbc.Row(skill_gauges, className='d-flex justify-content-center'), width=12),
        ], className='mb-4', style={'border': '1px solid #dee2e6', 'padding': '15px', 'background-color': '#f8f9fa'})

        sections.append(section)

    return sections


import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Function to create totals section divided into skill and non-skill parts
def create_totals_section(filtered_df):
    skill_totals = {}
    non_skill_totals = {}

    # Accumulate scores for skills and non-skills separately
    for _, row in filtered_df.iterrows():
        skill_passage = row['Skill/Passage']
        average_score = row['Average Score']
        percentage_score = average_score * 100

        # Remove prefixes from skill titles (e.g., 'A-SK-', 'B-SK-')
        clean_title = skill_passage.replace('A-SK-', '').replace('B-SK-', '').replace('C-SK-', '').replace('D-SK-', '') \
                                   .replace('A-', '').replace('B-', '').replace('C-', '').replace('D-', '')

        if '-SK-' in skill_passage:
            if clean_title in skill_totals:
                skill_totals[clean_title]['total_score'] += percentage_score
                skill_totals[clean_title]['count'] += 1
            else:
                skill_totals[clean_title] = {'total_score': percentage_score, 'count': 1}
        else:
            if clean_title in non_skill_totals:
                non_skill_totals[clean_title]['total_score'] += percentage_score
                non_skill_totals[clean_title]['count'] += 1
            else:
                non_skill_totals[clean_title] = {'total_score': percentage_score, 'count': 1}

    # Calculate the average score for each skill and non-skill
    avg_skill_scores = {title: data['total_score'] / data['count'] for title, data in skill_totals.items()}
    avg_non_skill_scores = {title: data['total_score'] / data['count'] for title, data in non_skill_totals.items()}

    # Determine min and max values to color accordingly
    min_skill_score = min(avg_skill_scores.values(), default=0)
    max_skill_score = max(avg_skill_scores.values(), default=100)
    min_non_skill_score = min(avg_non_skill_scores.values(), default=0)
    max_non_skill_score = max(avg_non_skill_scores.values(), default=100)

    # Create the gauges for the combined skills and non-skills
    skill_gauges = []
    non_skill_gauges = []

    for title, avg_score in avg_skill_scores.items():
        gauge_color = 'red' if avg_score == min_skill_score else 'green' if avg_score == max_skill_score else 'blue'
        
        total_gauge = dcc.Graph(
            id=f'gauge-{title}-skill-total',
            figure=go.Figure(go.Indicator(
                mode='gauge+number',
                value=avg_score,
                number={'font': {'size': 21, 'color': gauge_color}},  # 5% larger size
                title={
                    'text': wrap_text(f"{title}"),  # Clean title without "Total"
                    'font': {'size': 14}
                },
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color}
                }
            )),
            style={'display': 'inline-block', 'width': '320px', 'height': '320px', 'margin': '0px'}  # 5% larger size, 2px padding
        )

        skill_gauges.append(dbc.Col(total_gauge, width=3))

    for title, avg_score in avg_non_skill_scores.items():
        gauge_color = 'red' if avg_score == min_non_skill_score else 'green' if avg_score == max_non_skill_score else 'blue'
        
        total_gauge = dcc.Graph(
            id=f'gauge-{title}-non-skill-total',
            figure=go.Figure(go.Indicator(
                mode='gauge+number',
                value=avg_score,
                number={'font': {'size': 21, 'color': gauge_color}},  # 5% larger size
                title={
                    'text': wrap_text(f"{title}"),  # Clean title without "Total"
                    'font': {'size': 14}
                },
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color}
                }
            )),
            style={'display': 'inline-block', 'width': '320px', 'height': '320px', 'margin': '0px'}  # 5% larger size, 
        )

        non_skill_gauges.append(dbc.Col(total_gauge, width=3))

    # Split the gauges into two parts for skills and one part for non-skills
    half_index_skill = len(skill_gauges) // 2
    skill_part_1 = skill_gauges[:half_index_skill]
    skill_part_2 = skill_gauges[half_index_skill:]

    # Return the two parts for skills, a single part for non-skills, and a line separator
    return (
        dbc.Row(skill_part_1, className='d-flex justify-content-center', style={'margin-top': '15px'}),
        dbc.Row(skill_part_2, className='d-flex justify-content-center', style={'margin-top': '15px'}),
        html.Hr(style={'border': '1px solid #000', 'margin': '30px 0'}),
        dbc.Row(non_skill_gauges, className='d-flex justify-content-center', style={'margin-top': '15px'})
    )


# Updated callback to combine gauges with the same title
@app.callback(
    [Output('gauges-container', 'children'),
     Output('totals-container', 'children'),
     Output('test-version-dropdown', 'value')],
    [Input('student-search', 'value'),
     Input('test-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('test-version-dropdown', 'value')]
)
def update_gauges_and_totals(student_search, selected_versions, selected_countries, test_version):
    # Check if test_version is None, and if so, initialize it as an empty list
    if test_version is None:
        test_version = []

    # Handle "Select All" case for versions
    if 'ALL' in test_version:
        test_version = df['Version'].unique().tolist()

    filtered_df = df

    if student_search:
        filtered_df = filtered_df[filtered_df['Username'] == student_search]

    if selected_versions:
        filtered_df = filtered_df[filtered_df['Test'].isin(selected_versions)]

    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    if test_version:
        filtered_df = filtered_df[filtered_df['Version'].isin(test_version)]

    # Debugging: Check filtered data
    print(f"Filtered data rows: {len(filtered_df)}")

    if filtered_df.empty:
        return html.Div("No data available for selected filters."), html.Div(), test_version

    # Generate gauges and totals section
    gauges = create_gauge_sections(filtered_df)
    totals_parts = create_totals_section(filtered_df)

    # Return the two sections and the test version
    return gauges, html.Div(totals_parts), test_version

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
