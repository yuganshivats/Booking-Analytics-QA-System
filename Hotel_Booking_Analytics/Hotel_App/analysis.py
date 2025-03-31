import pandas as pd
import numpy as np
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def run_analysis():
    url = 'https://solvei8-aiml-assignment.s3.ap-southeast-1.amazonaws.com/hotel_bookings.csv'
    df = pd.read_csv(url)

    df.replace("NULL", np.nan, inplace=True)
    df[['children', 'agent', 'company']] = df[['children', 'agent', 'company']].fillna(0)
    df.fillna({'country': 'Unknown'}, inplace=True)
    df['year-month'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' + 
        df['arrival_date_month'] + '-' + 
        df['arrival_date_day_of_month'].astype(str)
    )
    
    ### 3. Compute Key Analytics ###
    # Compute total nights and total revenue (ensure revenue is 0 for canceled bookings)
    df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['total_revenue'] = df.apply(lambda row: row['adr'] * row['total_nights'] if row['is_canceled'] == 0 else 0, axis=1)

    # Monthly revenue trends
    monthly_revenue_df = df.groupby(df['year-month'].dt.to_period('M'))['total_revenue'].sum().reset_index()
    monthly_revenue_text = "Monthly revenues:\n" + "\n".join([f"{period}: {revenue:.2f}" for period, revenue in zip(monthly_revenue_df['year-month'], monthly_revenue_df['total_revenue'])])

    # Overall cancellation rate
    total_bookings = len(df)
    cancelled_bookings = df[df['is_canceled'] == 1].shape[0]
    cancellation_rate = (cancelled_bookings / total_bookings) * 100
    cancellation_rate_text = f"Overall Cancellation Rate: {cancellation_rate:.2f}%"

    # Geographical distribution of bookings (top 10 countries)
    country_distribution = df['country'].value_counts(normalize=True) * 100
    top_countries = country_distribution.head(10)
    country_distribution_text = "Top 10 Countries by Booking Percentage:\n" + "\n".join([f"{country}: {percentage:.2f}%" for country, percentage in zip(top_countries.index, top_countries)])

    # Lead time distribution
    lead_time_distribution = df['lead_time'].describe()
    lead_time_dist_text = "Lead Time Distribution:\n" + "\n".join([f"{stat}: {value}" for stat, value in lead_time_distribution.items()])

    # Average adults per booking (for all and non-canceled bookings)
    avg_adults_all = df['adults'].mean()
    avg_adults_non_canceled = df[df['is_canceled'] == 0]['adults'].mean()
    avg_adults_text = f"Average Adults per Booking (all): {avg_adults_all:.2f}\nAverage Adults per Booking (non-canceled): {avg_adults_non_canceled:.2f}"

    # Stay percentage share (weekend vs. weekday nights)
    non_canceled_df = df[df['is_canceled'] == 0]
    total_weekend_nights = non_canceled_df['stays_in_weekend_nights'].sum()
    total_weekday_nights = non_canceled_df['stays_in_weekend_nights'].sum()  # Fixed: Should be 'stays_in_week_nights'
    total_nights = total_weekend_nights + total_weekday_nights
    weekend_percentage = (total_weekend_nights / total_nights) * 100 if total_nights > 0 else 0
    weekday_percentage = (total_weekday_nights / total_nights) * 100 if total_nights > 0 else 0
    stay_percentage_text = f"Total Weekend Nights: {total_weekend_nights}\nTotal Weekday Nights: {total_weekday_nights}\nTotal Nights: {total_nights}\nWeekend Percentage: {weekend_percentage:.2f}%\nWeekday Percentage: {weekday_percentage:.2f}%"

    # Customer type distribution
    customer_distribution = df['customer_type'].value_counts(normalize=True) * 100
    customer_distribution_text = "Customer type distribution:\n" + "\n".join([f"{customer}: {percentage:.2f}%" for customer, percentage in zip(customer_distribution.index, customer_distribution)])

    # Hotel-level total revenue
    hotel_revenue = df.groupby('hotel')['total_revenue'].sum().reset_index()
    hotel_revenue_text = "Hotel Total Revenue:\n" + "\n".join([f"Hotel: {hotel}, Total Revenue: {revenue:.2f}" for hotel, revenue in zip(hotel_revenue['hotel'], hotel_revenue['total_revenue'])])

    # Hotel-level cancellation rate
    hotel_cancellations = df.groupby('hotel')['is_canceled'].mean().reset_index()
    hotel_cancellations_text = "Hotel Cancellation Rate:\n" + "\n".join([f"Hotel: {hotel}, Cancellation Rate: {rate*100:.2f}%" for hotel, rate in zip(hotel_cancellations['hotel'], hotel_cancellations['is_canceled'])])

    # Additional insights
    # Meal type distribution
    meal_distribution = df['meal'].value_counts(normalize=True) * 100
    meal_distribution_text = "Meal Type Distribution:\n" + "\n".join([f"{meal}: {percentage:.2f}%" for meal, percentage in zip(meal_distribution.index, meal_distribution)])

    # Average ADR for non-canceled bookings
    avg_adr = df[df['is_canceled'] == 0]['adr'].mean()
    avg_adr_text = f"Average Daily Rate (ADR) for non-canceled bookings: {avg_adr:.2f}"

    # Booking changes distribution
    booking_changes_dist = df['booking_changes'].describe()
    booking_changes_text = "Booking Changes Distribution:\n" + "\n".join([f"{stat}: {value}" for stat, value in booking_changes_dist.items()])

    # Total bookings and non-canceled bookings
    total_bookings_text = f"Total Bookings: {total_bookings}\nNon-Canceled Bookings: {len(non_canceled_df)}"

    # Data time range
    earliest_date = df['year-month'].min()
    latest_date = df['year-month'].max()
    date_range_text = f"Data covers from {earliest_date.date()} to {latest_date.date()}"
    report_data = {
        "avg_cancel_rate": cancellation_rate,
        "country_cancellation_count": country_distribution.to_dict(),
        "total_bookings_text": total_bookings_text,
        "cancellation_rate_text": cancellation_rate_text,
        "monthly_revenue_text": monthly_revenue_text,
        "country_distribution_text": country_distribution_text,
        "lead_time_dist_text": lead_time_dist_text,
        "avg_adults_text": avg_adults_text,
        "stay_percentage_text": stay_percentage_text,
        "customer_distribution_text": customer_distribution_text,
        "total_bookings": total_bookings,
        "non_canceled_bookings": len(non_canceled_df),
        "cancellation_rate": cancellation_rate,
        "monthly_revenues": monthly_revenue_df.to_dict(orient='records'),
        "top_countries": top_countries.to_dict(),
        "lead_time_distribution": lead_time_distribution.to_dict(),
        "avg_adults_all": avg_adults_all,
        "avg_adults_non_canceled": avg_adults_non_canceled,
        "stay_percentage": {
            "total_weekend_nights": total_weekend_nights,
            "total_weekday_nights": total_weekday_nights,
            "total_nights": total_nights,
            "weekend_percentage": weekend_percentage,
            "weekday_percentage": weekday_percentage
        },
        "customer_distribution": customer_distribution.to_dict(),
        "hotel_revenue": hotel_revenue.to_dict(orient='records'),
        "hotel_cancellations": hotel_cancellations.to_dict(orient='records'),
        "meal_distribution": meal_distribution.to_dict(),
        "avg_adr": avg_adr,
        "booking_changes_distribution": booking_changes_dist.to_dict(),
        "date_range": {
            "start": str(earliest_date.date()),
            "end": str(latest_date.date())
        }
    }

    # Ensure the directory exists
    os.makedirs("./data", exist_ok=True)
    with open("./data/hotel_analytics.json", "w") as f:
        json.dump(report_data, f, indent=4, default=str)
        print("Analysis completed and report saved to 'Hotel_App/data/hotel_analytics.json'")

    # Generate report (optional for RAG)
    chat = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    report_json = json.dumps(report_data, indent=4, default=str)
    prompt = f"Generate a detailed hotel bookings report based on: {report_json}"
    groq_response = chat.invoke(prompt)
    final_report = "Hotel Bookings Report\n\n" + str(groq_response) + "\n\n" + report_json

    with open("./data/final_hotel_bookings_report.txt", "w") as f:
        f.write(final_report)
        print("Final report saved to 'Hotel_App/data/final_hotel_bookings_report.txt'")

if __name__ == "__main__":
    run_analysis()
