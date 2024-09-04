import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

LOC_GRID_SIZE = 200


def basic_preprocess(df: pd.DataFrame, remove_estimated=True, can_drop_samples=True) -> pd.DataFrame:
    if can_drop_samples:
        # Remove empty rows
        df.dropna(inplace=True)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Remove rows where "door_closing_time" column is empty
        df = df[df['door_closing_time'].notna()]

    # Remove rows where "arrival_is_estimated" is "TRUE"
    if remove_estimated and can_drop_samples:
        df = df[df['door_closing_time'].notna()]
        df = df[df['arrival_is_estimated'] != "TRUE"]
    else:
        df['door_closing_time'] = df['door_closing_time'].fillna(df['arrival_time'], inplace=True)

    # Assign numerical labels to categorical columns
    label_encoder = LabelEncoder()
    df['part'] = label_encoder.fit_transform(df['part'])
    df['alternative'] = label_encoder.fit_transform(df['alternative'])
    df['cluster'] = label_encoder.fit_transform(df['cluster'])
    df['arrival_is_estimated'] = label_encoder.fit_transform(df['arrival_is_estimated'])

    # Split "arrival_time" into hour, minute, second
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df['arrival_minute'] = df['arrival_time'].dt.minute
    df['arrival_second'] = df['arrival_time'].dt.second

    # Split "door_closing_time" into hour, minute, second
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')
    df['closing_hour'] = df['door_closing_time'].dt.hour
    df['closing_minute'] = df['door_closing_time'].dt.minute
    df['closing_second'] = df['door_closing_time'].dt.second

    # Calculate total seconds since midnight for "arrival_time"
    df['arrival_total_seconds'] = df['arrival_hour'] * 3600 + df['arrival_minute'] * 60 + df['arrival_second']

    # Calculate total seconds since midnight for "door_closing_time"
    df['closing_total_seconds'] = df['closing_hour'] * 3600 + df['closing_minute'] * 60 + df['closing_second']

    # Delete the original "arrival_time" and "door_closing_time" columns
    df.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

    # Works when before and after midnight as well as after
    df['total_door_open_time'] = (df['closing_total_seconds'] - df['arrival_total_seconds'] + 3600 * 24) % (3600 * 24)

    if can_drop_samples:
        # if neg delete, if more than 10 minutes delete
        df = df[
            (df['total_door_open_time'] >= 0) & (df['total_door_open_time'] < 1000)]

    # Delete the "station_name" column if it exists
    if 'station_name' in df.columns:
        df.drop(columns=['station_name'], inplace=True)

    return df


def advanced_preprocess(df: pd.DataFrame, remove_estimated=True, can_drop_samples=True) -> pd.DataFrame:
    # Apply basic preprocessing first
    df = basic_preprocess(df, remove_estimated, can_drop_samples)

    # Feature Engineering: Peak hours
    peak_hours_morning = range(7, 10)  # 7:00 AM to 10:00 AM
    peak_hours_evening = range(16, 19)  # 4:00 PM to 7:00 PM

    df['is_peak_hour'] = df['arrival_hour'].isin(peak_hours_morning).astype(int) | df['arrival_hour'].isin(
        peak_hours_evening).astype(int)

    # Feature Engineering: Interaction terms
    df['station_direction_interaction'] = df['station_index'] * df['direction']
    df['latitude_longitude_interaction'] = df['latitude'] * df['longitude']

    # Feature Engineering: Lag features (previous stop's data)
    df = df.sort_values(by=['trip_id_unique', 'station_index'])
    df['previous_passengers_continue'] = df.groupby('trip_id_unique')['passengers_continue'].shift(1).fillna(0)
    df['previous_passengers_continue_menupach'] = df.groupby('trip_id_unique')['passengers_continue_menupach'].shift(
        1).fillna(0)

    # Feature Engineering: Difference in seconds between stops
    df['arrival_diff'] = df.groupby('trip_id_unique')['arrival_total_seconds'].diff().fillna(0)
    df['closing_diff'] = df.groupby('trip_id_unique')['closing_total_seconds'].diff().fillna(0)

    # Categorical Features: One-hot encoding for 'line_id', 'station_id', and 'cluster'
    categorical_features = ['line_id', 'station_id', 'cluster']
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(df[categorical_features])
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df.reset_index(drop=True), onehot_encoded_df.reset_index(drop=True)], axis=1)

    # Scaling numerical features
    scaler = StandardScaler()
    numerical_features = ['latitude', 'longitude', 'arrival_total_seconds', 'closing_total_seconds',
                          'total_door_open_time',
                          'previous_passengers_continue',
                          'previous_passengers_continue_menupach',
                          'arrival_diff', 'closing_diff']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    df = df.drop(['mekadem_nipuach_luz', 'trip_id_unique', 'trip_id_unique_station'] + ['closing_hour', 'closing_minute', 'closing_second',
                                      'closing_total_seconds', 'total_door_open_time'], axis=1, errors='ignore')
    return df.fillna(df.mean())
