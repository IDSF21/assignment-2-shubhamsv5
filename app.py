import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import seaborn as sns
import time
import argparse
import warnings

import matplotlib as mpl
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#Streamlit Page Configurations
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Global Warming: Climate Change vs Population</h1>", unsafe_allow_html=True)
st.text("")
st.text("")
st.text("")
st.text("")


#Ref:: Data Clean up and Animations taken from: https://www.kaggle.com/sevgisarac/climate-change/

def get_temp_change_data():
    temp_data = pd.read_csv("Environment_Temperature_Change.csv",
                            encoding='latin-1')
    countrycode_data = pd.read_csv('Country_Code.csv')
    temp_data.rename(columns={'Area': 'Country'}, inplace=True)
    temp_data = temp_data[temp_data['Element'] == 'Temperature change']
    countrycode_data.drop(['Country Code', 'M49 Code', 'ISO2 Code', 'Start Year', 'End Year'], axis=1, inplace=True)
    temp_data.drop(['Area Code', 'Months Code', 'Element Code', 'Element', 'Unit'], axis=1, inplace=True)
    countrycode_data.rename(columns={'ISO3 Code': 'Country Code'}, inplace=True)
    df = pd.merge(temp_data, countrycode_data, how='outer', on='Country')
    df = df.melt(id_vars=["Country Code", "Country", "Months", ], var_name="year", value_name="temp_change")
    df["year"] = [i.split("Y")[-1] for i in df.year]
    return df


def plot_temp_change():
    df = get_temp_change_data()
    df = df[df['Months'] == 'Meteorological year']  # chose yearly base data
    df['°C'] = ['<=-1.5' if x <= (-1.5) else '<=-1.0' if (-1.5) < x <= (-1.0) else '<=0.0' if (
                                                                                                  -1.0) < x <= 0.0 else '<=0.5' if 0.0 < x <= 0.5 else '<=1.5' if 0.5 < x <= 1.5 else '>1.5' if 1.5 <= x < 10 else 'None'
                for x in df['temp_change']]
    fig = px.choropleth(df, locations="Country Code",
                        color="°C",
                        locationmode='ISO-3',
                        hover_name="Country",
                        hover_data=['temp_change'],
                        animation_frame=df.year,
                        labels={'temp_change': 'The Temperature Change', '°C': '°C'},
                        category_orders={'°C': ['<=-1.5', '<=-1.0', '<=0.0', '<=0.5', '<=1.5', '>1.5', 'None']},
                        color_discrete_map={'<=-1.5': "#08519c", '<=-1.0': "#9ecae1", '<=0.0': "#eff3ff",
                                            '<=0.5': "#ffffb2", '<=1.5': "#fd8d3c", '>1.5': "#bd0026",
                                            'None': "#252525"})

    fig['layout'].update(
        width=1000,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        font_color="white",
        template='seaborn',
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            orientation="v",
            xanchor="right"
        ))
    return df, fig


col1, col2 = st.columns((3, 1))
df_temp_change, fig = plot_temp_change()

with col1:
    st.header("Temperature Change - 1961 - 2018")
    st.plotly_chart(fig)
with col2:
    st.header("Statistics")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(df_temp_change.describe().astype("object"))


# Ref: Referneced go Scatter code from documentations: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html
def plot_temperature_population(df, country):
    df = df[df["Country"] == country]
    fig1 = go.Scatter(
        x=df["Year"],
        y=df["AverageTemperature"],
        name="Average Temperature(°C)"
    )
    fig2 = go.Scatter(
        x=df["Year"],
        y=df["Population"],
        name='Population(millions)',
        yaxis='y2'
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)
    fig.add_trace(fig2, secondary_y=True)
    fig['layout'].update(height=600, width=800,
                         font_color="white",
                         template='seaborn',
                         plot_bgcolor="rgba(255, 255, 255, 255)"
                         )
    st.plotly_chart(fig)


def plot_min_temperature_population(df_min, country_name):
    plot_temperature_population(df_min, country_name)


def plot_max_temperature_population(df_max, country_name):
    plot_temperature_population(df_max, country_name)

def get_country_data():
    df_temperature = pd.read_csv("CleanTempData.csv")
    df_population = pd.read_csv("CountriesPopulation.csv")
    df_population["Population"] = df_population["Population"] / 1000000
    df_temperature["Year"] = df_temperature["dt"].apply(lambda row: row[0:4] if "-" in row else row[-4:]).astype("int")
    df_temperature_max = df_temperature.groupby(["Country", "Year"]).agg({"AverageTemperature": "max"}).reset_index()
    df_max = pd.merge(df_temperature_max, df_population, how='inner', left_on=['Country', 'Year'],
                      right_on=['Country', 'Year'])
    df_temperature_min = df_temperature.groupby(["Country", "Year"]).agg({"AverageTemperature": "min"}).reset_index()
    df_min = pd.merge(df_temperature_min, df_population, how='inner', left_on=['Country', 'Year'],
                      right_on=['Country', 'Year'])
    return df_min, df_max, df_max["Country"].unique()


def plot_temperature_world_population(df):
    fig1 = go.Scatter(
        x=df["Year"],
        y=df["AverageTemperature"],
        name="Average Temperature(°C)"
    )
    fig2 = go.Scatter(
        x=df["Year"],
        y=df["Population"],
        name='Population(billions)',
        yaxis='y2'
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)
    fig.add_trace(fig2, secondary_y=True)
    fig['layout'].update(height=600, width=800,
                         font_color="white",
                         template='seaborn',
                         plot_bgcolor="rgba(255, 255, 255, 255)"
                         )
    return df, fig


def world_temperature_population():
    df_temperature = pd.read_csv("CleanTempData.csv")
    df_population = pd.read_csv("World_Population.csv")
    df_population["World Population"] = df_population["World Population"].str.replace(",", "")
    df_population["Population"] = df_population["World Population"].astype("int") / 1000000000
    df_temperature["Year"] = df_temperature["dt"].apply(lambda row: row[0:4] if "-" in row else row[-4:]).astype("int")
    df_teamperature = df_temperature.groupby(["Year"]).agg({"AverageTemperature": "mean"}).reset_index()
    df = pd.merge(df_teamperature, df_population, how='inner', left_on=['Year'], right_on=['Year'])
    return plot_temperature_world_population(df)



col1, col2 = st.columns((2, 1))
df_temp_change, fig = world_temperature_population()

with col1:
    st.header("Average Temperature(World) vs World Population")
    st.plotly_chart(fig)
with col2:
    st.header("Statistics")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    df_temp_change.drop("Year",axis=1, inplace=True)
    st.write(df_temp_change.describe().astype("object"))
world_temperature_population()

st.text("")
st.text("")
st.text("")
st.text("")
plot_temp_change()
df_min_temp, df_max_temp, country_list = get_country_data()
country = st.selectbox("Select the Country:", country_list)
col1, col2 = st.columns(2)
col1.header("Min Average Temperature vs Total Population")
col2.header("Max Average Temperature vs Total Population")
with col1:
    plot_min_temperature_population(df_min_temp, country)
with col2:
    plot_max_temperature_population(df_max_temp, country)
