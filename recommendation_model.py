import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)

class PlaceRecommender:
    def __init__(self):
        self.places_df = None
        self.users_df = None
        self.ratings_df = None  # Add this line
        self.model = None
        self.place_id_map = {}
        self.attribute_cols = []
        self.mlb = MultiLabelBinarizer()
        self.user_encoder = LabelEncoder()
        self.load_and_process_data()
        self.setup_model()
        
    def load_and_process_data(self):
        # Load datasets
        self.places_df = pd.read_csv('final_places_with_attributes.csv')
        self.users_df = pd.read_csv('sample_new.csv')
        
        # Process data
        self.users_df['preferences'] = self.users_df['preferences'].apply(self.process_comma_separated_list)
        self.users_df['interactions'] = self.users_df['interactions'].apply(self.process_comma_separated_list)
        
        # Process place attributes
        place_attributes = self.mlb.fit_transform(self.places_df['Attributes'].apply(self.process_comma_separated_list))
        self.attribute_cols = self.mlb.classes_
        place_attributes_df = pd.DataFrame(place_attributes, columns=self.attribute_cols)
        
        # Combine with original places dataframe
        self.places_encoded_df = pd.concat([self.places_df[['place_id', 'place_name']], place_attributes_df], axis=1)
        
        # Encode user IDs
        self.users_df['encoded_user_id'] = self.user_encoder.fit_transform(self.users_df['user_id'])
        
        # Create place ID mapping
        self.place_id_map = {id_val: i for i, id_val in enumerate(self.places_encoded_df['place_id'].unique())}

    @staticmethod
    def process_comma_separated_list(text):
        if isinstance(text, str):
            return [item.strip() for item in text.split(',')]
        return []

    def setup_model(self):
        n_users = len(self.users_df)
        n_places = len(self.places_encoded_df)
        self.model = self.HybridRecommender(n_users, n_places, n_features=len(self.attribute_cols))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    class HybridRecommender(tf.keras.Model):
        def __init__(self, n_users, n_places, n_factors=64, n_features=None):
            super().__init__()
            self.user_embedding = tf.keras.layers.Embedding(n_users, n_factors)
            self.place_embedding = tf.keras.layers.Embedding(n_places, n_factors)
            self.feature_dense1 = tf.keras.layers.Dense(64, activation='relu')
            self.feature_dense2 = tf.keras.layers.Dense(n_factors, activation='relu')
            self.combine = tf.keras.layers.Concatenate(axis=1)
            self.dense1 = tf.keras.layers.Dense(32, activation='relu')
            self.dense2 = tf.keras.layers.Dense(16, activation='relu')
            self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        def call(self, inputs):
            user_id, place_id, features = inputs
            user_embedding = self.user_embedding(user_id)
            place_embedding = self.place_embedding(place_id)
            dot_product = tf.reduce_sum(tf.multiply(user_embedding, place_embedding), axis=1, keepdims=True)
            feature_vector = self.feature_dense1(features)
            feature_vector = self.feature_dense2(feature_vector)
            combined = self.combine([dot_product, feature_vector])
            x = self.dense1(combined)
            x = self.dense2(x)
            return self.output_layer(x)

    def train(self):
        # Generate interactions
        interactions = self.generate_interactions()
        train_data, _ = train_test_split(interactions, test_size=0.2, random_state=42)
        
        # Prepare batch data
        inputs, targets = self.prepare_batch_data(train_data)
        dataset = tf.data.Dataset.from_tensor_slices((tuple(inputs), targets)).shuffle(1000).batch(128)
        
        # Train model
        self.model.fit(dataset, epochs=20, verbose=1)
        self.model.save('place_recommender_model.keras')

    def generate_interactions(self):
        """
        Generate interactions from ratings data or fall back to preference-based interactions.
        """
        interactions = []
        
        # If we have ratings data, use it
        if self.ratings_df is not None and not self.ratings_df.empty:
            for _, row in self.ratings_df.iterrows():
                u_id = row['user_id']
                p_id = row['place_id']
                rating_val = row['rating_value']
                
                # Check if user is in our encoder
                try:
                    user_encoded = self.user_encoder.transform([u_id])[0]
                except:
                    # Skip ratings from users not in our training data
                    continue
                    
                interactions.append({
                    'user_id': user_encoded,
                    'place_id': p_id,
                    'rating': rating_val / 5.0  # Normalize rating to 0-1
                })
                
            if interactions:  # Only return if we found valid ratings
                return pd.DataFrame(interactions)
        
        # Fall back to preference-based interactions if no ratings data
        for _, user in self.users_df.iterrows():
            user_id = user['encoded_user_id']
            preferences = user['preferences']
            
            for _, place in self.places_encoded_df.iterrows():
                place_id = place['place_id']
                place_attrs = self.process_comma_separated_list(
                    self.places_df[self.places_df['place_id'] == place_id]['Attributes'].values[0]
                )
                match_score = self.calculate_preference_match(preferences, place_attrs)
                
                # Generate a rating based on preference match and whether user has interacted with this place
                rating = min(match_score + 0.5, 1.0) if place['place_name'] in user['interactions'] else max(
                    min(match_score + np.random.normal(0, 0.2), 1.0), 0.0
                )
                
                interactions.append({
                    'user_id': user_id,
                    'place_id': place_id,
                    'rating': rating
                })
                
        return pd.DataFrame(interactions)

    def prepare_batch_data(self, interactions_batch):
        user_ids = interactions_batch['user_id'].values
        place_ids = interactions_batch['place_id'].values
        place_indices = np.array([self.place_id_map.get(pid, 0) for pid in place_ids])
        features = np.zeros((len(place_ids), len(self.attribute_cols)))
        
        for i, pid in enumerate(place_ids):
            place_row = self.places_encoded_df[self.places_encoded_df['place_id'] == pid]
            if not place_row.empty:
                features[i] = place_row[self.attribute_cols].values[0]
        
        return [user_ids, place_indices, features], interactions_batch['rating'].values

    @staticmethod
    def calculate_preference_match(user_prefs, place_attrs):
        return min(sum(1 for pref in user_prefs if pref in place_attrs) / max(len(user_prefs), 1), 1.0)

    def recommend(self, user_id, top_n=5):
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        if user_row.empty:
            return "User not found"

        encoded_user_id = user_row['encoded_user_id'].values[0]
        user_prefs = user_row['preferences'].values[0]

        # Get matching places
        matching_places = self.get_matching_places(user_prefs)
        if matching_places.empty:
            return "No places match the user's preferences."

        # Get recommendations
        return self.get_top_recommendations(matching_places, encoded_user_id, user_prefs, top_n)

    def get_matching_places(self, user_prefs):
        return self.places_encoded_df[self.places_encoded_df.apply(
            lambda row: any(pref in self.process_comma_separated_list(
                self.places_df[self.places_df['place_id'] == row['place_id']]['Attributes'].values[0]
            ) for pref in user_prefs),
            axis=1
        )].copy()

    def get_top_recommendations(self, matching_places, encoded_user_id, user_prefs, top_n):
        # Prepare prediction data
        place_ids = matching_places['place_id'].values
        place_indices = np.array([self.place_id_map.get(pid, 0) for pid in place_ids])
        features = matching_places[self.attribute_cols].values
        user_ids = np.full(len(place_ids), encoded_user_id)
        
        # Make predictions
        predictions = self.model([user_ids, place_indices, features]).numpy().flatten()
        matching_places.loc[:, 'predictions'] = predictions
        
        # Calculate matched preferences
        matching_places['matched_prefs_count'] = matching_places.apply(
            lambda row: sum(pref in self.process_comma_separated_list(
                self.places_df[self.places_df['place_id'] == row['place_id']]['Attributes'].values[0]
            ) for pref in user_prefs),
            axis=1
        )
        
        # Sort and get top recommendations
        matching_places = matching_places.sort_values(
            by=['matched_prefs_count', 'predictions'], 
            ascending=[False, False]
        )
        
        return self.format_recommendations(matching_places[:top_n], user_prefs)

    def format_recommendations(self, top_places, user_prefs):
        result = []
        for _, place in top_places.iterrows():
            place_attrs = self.process_comma_separated_list(
                self.places_df[self.places_df['place_id'] == place['place_id']]['Attributes'].values[0]
            )
            matching_prefs = [pref for pref in user_prefs if pref in place_attrs]
            result.append({
                'place_id': place['place_id'],  # Make sure to include this
                'place_name': place['place_name'],
                'matching_preferences': matching_prefs,
                'attributes': place_attrs,
                'match_score': place.get('predictions', 0.5)
            })
        return result

# Create global recommender instance
recommender = PlaceRecommender()

def get_db_connection():
    return create_engine('postgresql://postgres:123@localhost/postgres')

def recommend_places_for_user(user_id, top_n=10):
    try:
        # Get fresh data from database
        engine = get_db_connection()
        users_df = pd.read_sql(f"SELECT * FROM users WHERE id = {user_id}", engine)
        
        if users_df.empty:
            return "User not found"

        user_prefs = users_df['preferences'].iloc[0].split(',') if users_df['preferences'].iloc[0] else []
        
        if not user_prefs:
            return "No preferences set for this user"

        # Get all places
        places_df = pd.read_sql("SELECT * FROM places", engine)
        
        # Get ratings
        ratings_df = pd.read_sql("SELECT * FROM ratings", engine)
        
        # Process ALL places, not just ones with matching preferences
        all_places = []
        for _, place in places_df.iterrows():
            place_attrs = place['attributes'].split(',') if place['attributes'] else []
            matching_prefs = [pref for pref in user_prefs if pref in place_attrs]
            
            # Calculate match score
            match_score = len(matching_prefs) / len(user_prefs) if user_prefs else 0
            
            # Check if user has rated this place
            user_rating = None
            if not ratings_df.empty:
                user_ratings = ratings_df[
                    (ratings_df['user_id'] == user_id) & 
                    (ratings_df['place_id'] == place['place_id'])
                ]
                if not user_ratings.empty:
                    user_rating = user_ratings['rating_value'].iloc[0]
            
            # Include ALL places in our collection
            all_places.append({
                'place_id': place['place_id'],
                'place_name': place['place_name'],
                'matching_preferences': matching_prefs,
                'attributes': place_attrs,
                'match_score': match_score,
                'user_rating': user_rating,  # This comes from the ratings table
                'description': place['description'] if 'description' in place else None
            })
        
        if not all_places:
            return "No places available in the database"
            
        # FIRST APPROACH: Try to get at least one place for EACH preference
        # Group places by which preference they match
        preference_to_places = {pref: [] for pref in user_prefs}
        
        for place in all_places:
            for pref in place['matching_preferences']:
                if pref in preference_to_places:
                    preference_to_places[pref].append(place)
        
        # For each preference, select the best-matching place
        selected_places = []
        covered_prefs = set()
        
        # Try to get one place for each preference
        for pref in user_prefs:
            matching = preference_to_places[pref]
            if matching:
                # Sort by match score for this preference
                best_match = sorted(matching, key=lambda x: x['match_score'], reverse=True)[0]
                
                # Only add if we haven't already selected this place
                if best_match not in selected_places:
                    selected_places.append(best_match)
                    covered_prefs.add(pref)
        
        # Now fill remaining slots (if any) with other high-scoring places
        remaining_slots = top_n - len(selected_places)
        if remaining_slots > 0:
            # Filter out places we've already selected
            remaining_places = [p for p in all_places if p not in selected_places and p['matching_preferences']]
            
            # Sort by match score, descending
            remaining_places.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Add until we reach top_n
            selected_places.extend(remaining_places[:remaining_slots])
        
        return selected_places
        
    except Exception as e:
        print(f"Error in recommend_places_for_user: {e}")
        return []

def retrain_model():
    """Function to retrain the model with updated ratings data"""
    try:
        # Refresh data from database before retraining
        engine = get_db_connection()
        new_places_df = pd.read_sql("SELECT * FROM places", engine)
        new_users_df = pd.read_sql("SELECT * FROM users", engine)
        ratings_df = pd.read_sql("SELECT * FROM ratings", engine)
        
        # Update recommender data
        if not new_places_df.empty and not new_users_df.empty:
            recommender.places_df = new_places_df
            recommender.users_df = new_users_df
            recommender.ratings_df = ratings_df  # Add this line
            recommender.load_and_process_data()
        
        # Only retrain if we have ratings
        if ratings_df.empty:
            print("No ratings data available for training")
            return False
            
        # Retrain the model
        recommender.train()
        print("Model retrained successfully with ratings data")
        return True
    except Exception as e:
        print(f"Error retraining model: {e}")
        return False