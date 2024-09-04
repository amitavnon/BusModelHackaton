from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import preprocess_data


# Load the data
def load_set(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="ISO-8859-8")


def get_color(passengers_up):
    if 0 <= passengers_up <= 5:
        return 'green'
    elif 6 <= passengers_up <= 50:
        return 'blue'
    elif 51 <= passengers_up <= 200:
        return 'pink'
    elif 201 <= passengers_up <= 300:
        return 'purple'
    elif 301 <= passengers_up <= 500:
        return 'orange'
    else:
        return 'red'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    args = parser.parse_args()

    orig_data = load_set(args.training_set)
    data = preprocess_data.basic_preprocess(orig_data)

    # Aggregate data to get total number of buses and total passengers up per hour
    buses_per_hour = data.groupby('arrival_hour')['trip_id'].count().reset_index(name='total_buses')
    passengers_per_hour = data.groupby('arrival_hour')[
        'passengers_up'].sum().reset_index(name='total_passengers_up')
    # Merge the two DataFrames
    merged_data = pd.merge(buses_per_hour, passengers_per_hour,
                           on='arrival_hour')
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # Bar plot for total buses
    ax1.bar(merged_data['arrival_hour'] - 0.2, merged_data['total_buses'],
            width=0.4, label='Total Buses', align='center')
    # Bar plot for total passengers up
    ax1.bar(merged_data['arrival_hour'] + 0.2,
            merged_data['total_passengers_up'], width=0.4,
            label='Total Passengers Up', align='center')
    # Add labels, title and legend
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Count')
    ax1.set_title('Total Number of Buses and Total Passengers Up per Hour')
    ax1.legend()
    plt.grid(True)
    plt.savefig('Total Number of Buses and Total Passengers Up per Hour.png')
    plt.close()

#############################################################################
    # Aggregate the data to get the total number of passengers up for each hour
    passengers_up_by_hour = data.groupby('arrival_hour')[
        'passengers_up'].sum().reset_index()

    # Create a bar plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(data=passengers_up_by_hour, x='arrival_hour',
                y='passengers_up', palette='viridis')
    plt.title('Total Passengers Up vs. Time of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Passengers Up')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('Total Passengers Up vs. Time of Day.png')
    plt.close()


###############################################################################

    # Scatter plot of longitude and latitude, colored and sized by 'passenger_up'
    station_data = data.groupby('station_id').agg({
        'longitude': 'first',
        'latitude': 'first',
        'passengers_up': 'sum'})

    station_data['color'] = station_data['passengers_up'].apply(get_color)
    plt.figure(figsize=(20, 20))
    plt.scatter(station_data['longitude'], station_data['latitude'],
                c=station_data['color'].tolist(),
                alpha=0.6)
    plt.title('Bus Stops by Longitude and Latitude', fontsize=20)
    plt.xlabel('Longitude', fontsize=16)
    plt.ylabel('Latitude', fontsize=16)
    plt.grid(True)

    # Create legend for color mapping
    legend_labels = {
        'green': '0-5 passengers_up',
        'blue': '6-50 passengers_up',
        'pink': '51-200 passengers_up',
        'purple': '201-300 passengers_up',
        'orange': '301-500 passengers_up',
        'red': ' >500 passengers_up'
    }
    handles = [plt.Line2D([], [], marker='o', color=color, linestyle='None',
                          markersize=10, label=label) for
               color, label in legend_labels.items()]
    plt.legend(handles=handles, title='Passenger Count',
               title_fontsize='large', fontsize='large', loc='upper right')

    plt.savefig(
        'Bus Stops by Longitude and Latitude.png')  # Save Seaborn plot as PNG

    ############################################################
    # look per line how many passengers are there in total
    total_passengers = data.groupby('line_id')['passengers_up'].sum().reset_index()
    unique_trips = data.groupby('line_id')['trip_id_unique'].nunique().reset_index()
    line_passengers = pd.merge(total_passengers, unique_trips, on='line_id')
    line_passengers.columns = ['line_id', 'total_passengers_up', 'unique_trips']
    line_passengers['average_passengers_per_trip'] = line_passengers['total_passengers_up'] / line_passengers['unique_trips']
    line_passengers = line_passengers.sort_values(by='average_passengers_per_trip')
    line_passengers.to_csv('line_passengers_per_trip_sorted.csv', index=False)

    ############################################################

    # Define the path to your CSV file
    csv_path = 'line_passengers_per_trip_sorted.csv'

    # Load data using the provided function
    line_passengers = load_set(csv_path)

    # Get the top 5 and bottom 5 bus lines by average passengers per trip
    top_5_lines = line_passengers.nlargest(5, 'average_passengers_per_trip')
    bottom_5_lines = line_passengers.nsmallest(5, 'average_passengers_per_trip')

    # Concatenate the top and bottom 5 lines
    lines_to_plot = pd.concat([top_5_lines, bottom_5_lines])

    # Create a numerical sequence for x-axis positions
    x_positions = range(len(lines_to_plot))

    # Plotting with color differentiation and rotated x-axis labels
    plt.figure(figsize=(12, 8))  # Adjust figure size as needed
    bars = plt.bar(x_positions, lines_to_plot['average_passengers_per_trip'],
                   tick_label=lines_to_plot['line_id'])

    # Color the top 5 bars in red and bottom 5 bars in green
    for i in range(5):
        bars[i].set_color('red')  # Top 5 lines
        bars[-(i + 1)].set_color('green')  # Bottom 5 lines

    plt.xlabel('Bus Line ID')
    plt.ylabel('Average Passengers per Trip')
    plt.title('Top 5 and Bottom 5 Bus Lines by Average Passengers per Trip')
    plt.xticks(x_positions, lines_to_plot['line_id'], rotation=45, ha='right')  # Set x-axis positions and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig("Top 5 and Bottom 5 Bus Lines by Average Passengers per Trip.png")
