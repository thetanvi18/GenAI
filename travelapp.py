import streamlit as st

# Title and Introduction
st.set_page_config(page_title="Personalized Travel Itinerary Planner", layout="wide")
st.title("🌍 Personalized Travel Itinerary Planner")
st.markdown("Welcome! This app helps you create a detailed and personalized travel itinerary based on your preferences.")

# User Input Form
st.header("Tell Us About Your Trip")

# Collecting User Inputs
budget = st.selectbox("What's your budget for this trip?", ["Low", "Moderate", "High"])
trip_duration = st.number_input("How many days will your trip last?", min_value=1, max_value=30, step=1)
destinations = st.text_input("Enter your destinations (comma-separated):")
purpose = st.selectbox("What's the purpose of your travel?", ["Leisure", "Adventure", "Cultural Exploration", "Food Tourism", "Business + Leisure"])
preferences = st.multiselect("What are your preferences?", ["Museums", "Nature", "Adventure", "Nightlife", "Hidden Gems", "Local Food"])
dietary = st.text_input("Do you have any dietary restrictions or preferences? (e.g., vegetarian, vegan)")
mobility = st.radio("Do you have any mobility concerns?", ["No", "Yes, minimal walking preferred"])
accommodation = st.selectbox("What type of accommodation do you prefer?", ["Luxury", "Budget", "Central Location"])

# Generate Itinerary Button
if st.button("Generate Itinerary"):
    if not destinations:
        st.error("Please enter at least one destination.")
    else:
        st.success("Your itinerary is ready! Enjoy your trip!")

st.sidebar.markdown("### About")
st.sidebar.info("This app generates a personalized travel itinerary.")
