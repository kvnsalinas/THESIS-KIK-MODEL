import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from recommendation_model import recommend_places_for_user, retrain_model
from recommendation_model import get_db_connection, recommend_places_for_user
from dotenv import load_dotenv


load_dotenv()  # Load environment variables

app = Flask(__name__)

# Configure the app using environment variables
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or "postgresql://postgres:123@localhost/postgres"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', "supersecretkey")

# Initialize SQLAlchemy with the app
db = SQLAlchemy(app)

class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    preferences = db.Column(db.Text, nullable=True, default="")
    interactions = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    role = db.Column(db.Integer, default=0)

class Places(db.Model):
    __tablename__ = 'places'
    place_id = db.Column(db.Integer, primary_key=True)
    place_name = db.Column(db.String(100), nullable=False)
    attributes = db.Column(db.Text, nullable=True)
    
    @classmethod
    def get_all_places(cls):
        return cls.query.all()

class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    username = db.Column(db.String(100), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text, nullable=True)

class Ratings(db.Model):
    __tablename__ = 'ratings'
    rating_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    place_id = db.Column(db.Integer, db.ForeignKey('places.place_id'), nullable=False)
    rating_value = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session:
            flash("Admin access required.", "error")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route("/")
def index():
    # Remove any redirections - just show index.html
    return render_template("index.html")

@app.route('/profile')
@login_required
def profile_page():
    user = Users.query.get(session["user_id"])
    if not user:
        flash("User not found!", "error")
        return redirect(url_for("login_page"))        
    
    preferences = user.preferences.split(",") if user.preferences else []
    user_data = {
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "joined_date": user.created_at.strftime("%B %d, %Y"),
        "last_login": user.last_login.strftime("%B %d, %Y %I:%M %p") if user.last_login else "Never",
        "preferences": preferences
    }
    return render_template('profile.html', user=user_data)

@app.route("/login_page")
def login_page():
    return render_template("login.html")

@app.route('/signup_page')
def signup_page():
    return render_template('signup.html')  # Make sure this exists

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validate all required fields
        if not all([first_name, last_name, email, username, password, confirm_password]):
            flash("All fields are required!", "error")
            return redirect(url_for("signup_page"))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup_page"))

        # Check for existing username or email
        existing_user = Users.query.filter(
            (Users.username == username) | (Users.email == email)
        ).first()
        
        if existing_user:
            if existing_user.username == username:
                flash("Username already exists!", "error")
            else:
                flash("Email already registered!", "error")
            return redirect(url_for("signup_page"))

        try:
            hashed_password = generate_password_hash(password, method='scrypt')
            new_user = Users(
                first_name=first_name,
                last_name=last_name,
                email=email,
                username=username,
                password=hashed_password,
                role=0,
                created_at=datetime.utcnow(),
                is_active=True
            )
            db.session.add(new_user)
            
            # Add audit log for new user creation
            audit_log = AuditLog(
                username=username,
                action="Account Creation",
                details=f"New user account created: {username} (Email: {email})"
            )
            db.session.add(audit_log)
            
            db.session.commit()
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login_page"))
        except Exception as e:
            print(f"Error creating user: {e}")
            db.session.rollback()
            flash("An error occurred. Please try again.", "error")
            return redirect(url_for("signup_page"))

    # GET request - render signup page
    return render_template('signup.html')

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    if not username or not password:
        # Log incomplete login attempt
        audit_log = AuditLog(
            username=username or "Unknown",
            action="Failed Login Attempt",
            details="Incomplete login credentials provided"
        )
        db.session.add(audit_log)
        db.session.commit()
        flash("Please enter both username and password.", "error")
        return redirect(url_for("login_page"))

    user = Users.query.filter_by(username=username).first()

    if user and check_password_hash(user.password, password):
        if not user.is_active:
            # Log inactive account login attempt
            audit_log = AuditLog(
                username=username,
                action="Failed Login Attempt",
                details="Attempt to login to inactive account"
            )
            db.session.add(audit_log)
            db.session.commit()
            return redirect(url_for('inactive_account'))

        session["user_id"] = user.id
        session["username"] = user.username
        session["role"] = user.role
        user.last_login = datetime.utcnow()
        
        # Log successful login
        audit_log = AuditLog(
            username=username,
            action="Successful Login",
            details=f"User logged in successfully from IP: {request.remote_addr}"
        )
        db.session.add(audit_log)
        db.session.commit()
        
        flash(f"Welcome back, {username}!", "success")
        return redirect(url_for("main_page"))
    
    # Log failed login attempt
    audit_log = AuditLog(
        username=username,
        action="Failed Login Attempt",
        details=f"Invalid credentials used from IP: {request.remote_addr}"
    )
    db.session.add(audit_log)
    db.session.commit()
    
    flash("Invalid username or password.", "error")
    return redirect(url_for("login_page"))

@app.route("/main")
@login_required
def main_page():
    if "user_id" in session:
        return render_template("main.html", username=session["username"])
    return redirect(url_for("login_page"))

@app.route("/preferences", methods=["GET", "POST"])
@login_required
def preferences_page():
    user = Users.query.get(session["user_id"])
    preference_options = {
        "Meals": ["Breakfast", "Lunch", "Dinner"],
        "Beverages": ["Coffee", "Alcohol Served"],
        "Cuisines": ["Filipino Cuisine", "Japanese Cuisine", "Seafood", "Italian Cuisine"],
        "Amenities": ["Scenic View", "Outdoor Seating", "PWD Friendly", "24 Hours"],
        "Services": ["Dine In", "Delivery Available", "Takeout Available"],
        "Accommodations": ["Resort", "Hotel", "Camping Site", "Hostel"],
        "Activities": ["Hiking", "Swimming", "Camping"]
    }

    user_preferences = user.preferences.split(",") if user.preferences else []

    if request.method == "POST":
        selected_preferences = request.form.getlist("preferences")
        user.preferences = ",".join(selected_preferences)
        
        # Enhanced audit log for preferences update
        audit_log = AuditLog(
            username=user.username,
            action="Preferences Updated",
            details=f"User preferences changed to: {', '.join(selected_preferences)}"
        )
        db.session.add(audit_log)
        db.session.commit()
        
        retrain_model()
        flash("Preferences updated successfully!", "success")
        return redirect(url_for("main_page"))

    return render_template("preferences.html",
                         preference_options=preference_options,
                         user_preferences=user_preferences)

@app.route("/logout")
def logout():
    if "username" in session:
        # Add audit log for logout
        audit_log = AuditLog(
            username=session["username"],
            action="Logout",
            details="User logged out"
        )
        db.session.add(audit_log)
        db.session.commit()
    
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for("index"))

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if "admin_id" in session:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        admin = Users.query.filter_by(username=username, role=1).first()
        
        if admin and check_password_hash(admin.password, password):
            session["admin_id"] = admin.id
            session["admin_username"] = admin.username
            flash("Login successful!", "success")
            return redirect(url_for("admin_dashboard"))
        
        flash("Invalid credentials", "error")
    
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    admin_username = session.get("admin_username", "Admin")
    stats = {
        "total_users": Users.query.count(),
        "active_users": Users.query.filter_by(is_active=True).count(),
        "inactive_users": Users.query.filter_by(is_active=False).count()
    }
    users = Users.query.order_by(Users.created_at.desc()).all()
    audit_logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(50).all()
    return render_template("admin_dashboard.html",
                         stats=stats, 
                         users=users,
                         audit_logs=audit_logs,
                         admin_username=admin_username)

@app.route("/admin/edit_user/<int:user_id>", methods=["GET", "POST"])
@admin_required
def edit_user(user_id):
    user = Users.query.get_or_404(user_id)
    if request.method == "POST":
        # Store old values for audit log
        old_values = {
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "is_active": user.is_active
        }
        
        # Update user
        user.first_name = request.form.get("first_name")
        user.last_name = request.form.get("last_name")
        user.email = request.form.get("email")
        user.is_active = bool(request.form.get("is_active"))
        
        # Create audit log entry
        changes = []
        if old_values["first_name"] != user.first_name:
            changes.append(f"first name: {old_values['first_name']} → {user.first_name}")
        if old_values["last_name"] != user.last_name:
            changes.append(f"last name: {old_values['last_name']} → {user.last_name}")
        if old_values["email"] != user.email:
            changes.append(f"email: {old_values['email']} → {user.email}")
        if old_values["is_active"] != user.is_active:
            changes.append(f"status: {'Active' if old_values['is_active'] else 'Inactive'} → {'Active' if user.is_active else 'Inactive'}")
        
        # Enhanced audit log for admin user edit
        audit_log = AuditLog(
            username=session.get("admin_username"),
            action="Admin User Edit",
            details=f"Admin modified user {user.username} - Changes: {', '.join(changes)}"
        )
        db.session.add(audit_log)
        db.session.commit()
        
        flash("User updated successfully!", "success")
        return redirect(url_for("admin_dashboard"))
    return render_template("edit_user.html", user=user)

@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    if user_id == session.get("admin_id"):
        flash("Cannot delete your own admin account!", "error")
        return redirect(url_for("admin_dashboard"))
    
    user = Users.query.get_or_404(user_id)
    
    # Enhanced audit log for user deletion
    audit_log = AuditLog(
        username=session.get("admin_username"),
        action="Admin User Delete",
        details=f"Admin deleted user: {user.username} (ID: {user.id}, Email: {user.email})"
    )
    db.session.add(audit_log)
    
    # Delete user
    db.session.delete(user)
    db.session.commit()
    
    flash("User deleted successfully!", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/logout")
@admin_required
def admin_logout():
    session.pop("admin_id", None)
    session.pop("admin_username", None)
    flash("Successfully logged out from admin panel", "success")
    return redirect(url_for("admin_login"))

@app.route("/recommend")
@login_required
def recommend_page():
    user_id = session["user_id"]
    user = Users.query.get(user_id)
    recommendations = recommend_places_for_user(user_id, top_n=10)
    
    # Add audit log for recommendation generation
    audit_log = AuditLog(
        username=user.username,
        action="Generated Recommendations",
        details=f"User generated place recommendations ({len(recommendations)} places)"
    )
    db.session.add(audit_log)
    db.session.commit()
    
    # Get user preferences directly from the database model
    user_prefs = []
    try:
        if user and user.preferences:
            user_prefs = user.preferences.split(',')
    except Exception as e:
        print(f"Error getting user preferences: {e}")
    
    # Make sure we have a notification element in recommend.html
    return render_template("recommend.html", 
                          recommendations=recommendations, 
                          user_prefs=user_prefs)

@app.route("/like_place/<place_name>", methods=["POST"])
@login_required
def like_place(place_name):
    try:
        user = Users.query.get(session["user_id"])
        
        # Initialize interactions as empty string if None
        if user.interactions is None:
            user.interactions = ""
            
        current_interactions = user.interactions.split(",") if user.interactions else []
        
        # Clean empty strings from the list
        current_interactions = [i for i in current_interactions if i]
        
        if place_name not in current_interactions:
            current_interactions.append(place_name)
            user.interactions = ",".join(current_interactions)
            db.session.commit()
            
            # Only retrain model if we have enough new interactions
            if len(current_interactions) % 5 == 0:  # Retrain every 5 new interactions
                try:
                    retrain_model()
                    print("Model retrained after new interactions")
                except Exception as e:
                    print(f"Model retraining failed: {e}")
                    
            return jsonify({"success": True, "message": "Place added to favorites!"})
        
        return jsonify({"success": False, "message": "Already in favorites"}), 400
    except Exception as e:
        print(f"Error in like_place: {e}")
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/places")
@login_required
def places_page():
    return render_template("places.html")

@app.route("/update_profile", methods=["POST"])
@login_required
def update_profile():
    user = Users.query.get(session["user_id"])
    if not user:
        flash("User not found!", "error")
        return redirect(url_for("login_page"))
    
    try:
        # Store old values for audit log
        old_values = {
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email
        }
        
        # Update user details
        user.first_name = request.form.get("first_name")
        user.last_name = request.form.get("last_name")
        user.email = request.form.get("email")
        
        changes = []
        if old_values["first_name"] != user.first_name:
            changes.append(f"First Name: {old_values['first_name']} → {user.first_name}")
        if old_values["last_name"] != user.last_name:
            changes.append(f"Last Name: {old_values['last_name']} → {user.last_name}")
        if old_values["email"] != user.email:
            changes.append(f"Email: {old_values['email']} → {user.email}")
        
        if changes:
            # Enhanced audit log for profile updates
            audit_log = AuditLog(
                username=user.username,
                action="Profile Update",
                details=f"User updated profile - Changes: {'; '.join(changes)}"
            )
            db.session.add(audit_log)
            
        db.session.commit()
        flash("Profile updated successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash("An error occurred while updating profile.", "error")
        print(f"Error updating profile: {e}")
    
    return redirect(url_for("profile_page"))

@app.route("/preferences_check")
@login_required
def check_preferences():
    user = Users.query.get(session["user_id"])
    has_preferences = bool(user.preferences)
    
    # Add audit log for attempted recommendation access without preferences
    if not has_preferences:
        audit_log = AuditLog(
            username=user.username,
            action="Recommendation Access Attempt",
            details="User attempted to access recommendations without setting preferences"
        )
        db.session.add(audit_log)
        db.session.commit()
    
    return jsonify({"has_preferences": has_preferences})

@app.route('/inactive-account')
def inactive_account():
    return render_template('inactive_account.html')

@app.route("/admin/clear_logs", methods=["POST"])
@admin_required
def clear_logs():
    try:
        # Instead of just deleting, drop and recreate the table
        db.session.execute('TRUNCATE TABLE audit_logs RESTART IDENTITY')
        
        # Add a new log entry for the clearing action
        new_log = AuditLog(
            username=session.get("admin_username"),
            action="Clear Logs",
            details="All audit logs were cleared"
        )
        db.session.add(new_log)
        db.session.commit()
        
        flash("Audit logs cleared successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash("Error clearing audit logs.", "error")
        print(f"Error clearing logs: {e}")
    
    return redirect(url_for("admin_dashboard"))

@app.route("/rate_place/<int:place_id>", methods=["POST"])
@login_required
def rate_place(place_id):
    try:
        user_id = session["user_id"]
        rating_val = request.form.get("rating")

        if not rating_val:
            return jsonify({"success": False, "message": "No rating provided"}), 400

        rating_val = int(rating_val)
        if rating_val < 1 or rating_val > 5:
            return jsonify({"success": False, "message": "Rating must be 1-5"}), 400
            
        user = Users.query.get(user_id)
        place = Places.query.get(place_id)
        
        if not place:
            return jsonify({"success": False, "message": "Place not found"}), 404

        # Check if user rated this place before
        existing_rating = Ratings.query.filter_by(user_id=user_id, place_id=place_id).first()
        if existing_rating:
            existing_rating.rating_value = rating_val
            existing_rating.created_at = datetime.utcnow()
            action = "Rating Updated"
            msg = "Your rating has been updated!"
        else:
            # Create new rating
            new_rating = Ratings(
                user_id=user_id,
                place_id=place_id,
                rating_value=rating_val,
                created_at=datetime.utcnow()
            )
            db.session.add(new_rating)
            action = "Rating Added"
            msg = "Your rating has been recorded!"
            
        # HYBRID APPROACH: Update interactions when a user rates a place
        # Initialize interactions as empty string if None
        if user.interactions is None:
            user.interactions = ""
            
        current_interactions = user.interactions.split(",") if user.interactions else []
        current_interactions = [i for i in current_interactions if i]  # Remove empty strings
        
        # Add place to interactions if not already there
        if place.place_name not in current_interactions:
            current_interactions.append(place.place_name)
            user.interactions = ",".join(current_interactions)
            
            # Add to audit log that we updated interactions based on rating
            details_msg = f"User rated place '{place.place_name}' as {rating_val} stars. Added to interactions automatically."
        else:
            details_msg = f"User rated place '{place.place_name}' as {rating_val} stars."

        # Log action in audit logs
        audit_entry = AuditLog(
            username=user.username,
            action=action,
            details=details_msg
        )
        db.session.add(audit_entry)
        db.session.commit()
        
        # Consider retraining the model if this is a significant update
        try:
            retrain_model()
        except Exception as e:
            print(f"Model retraining failed after rating: {e}")

        return jsonify({"success": True, "message": msg}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        try:
            admin = Users.query.filter_by(username="admin", role=1).first()
            if not admin:
                admin_password = "admin123"
                hashed_password = generate_password_hash(admin_password, method='scrypt')
                admin = Users(
                    username="admin",
                    password=hashed_password,
                    first_name="Admin",
                    last_name="User",
                    email="admin@travelmate.com",
                    role=1,
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.session.add(admin)
                db.session.commit()
                print("\n=== Admin Account Created ===")
                print(f"Username: admin")
                print(f"Password: {admin_password}")
                print(f"Email: admin@travelmate.com")
        except Exception as e:
            print(f"Error setting up admin account: {e}")
            db.session.rollback()

    app.run(debug=True)
