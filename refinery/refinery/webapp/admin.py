from flask import g, redirect, render_template, session, url_for, flash #url_for, abort, render_template, flash, send_from_directory, jsonify, Response, json
from refinery import app, lm
from flask.ext.login import login_user, logout_user, login_required
from refinery.data.models import User
from flask_wtf import Form
from wtforms import TextField, PasswordField, BooleanField
from wtforms.validators import DataRequired
from refinery.data.models import User

class LoginForm(Form):
    '''
    A form used in the main login page
    '''
    username = TextField('username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('remember_me', default = False)
    def __init__(self, *args, **kwargs):
        Form.__init__(self, *args, **kwargs)
        self.user = None

    def validate(self):
        rv = Form.validate(self)
        if not rv:
            print "not successful"
            return False

        user = User.query.filter_by( username=self.username.data ).first()
        if user is None:
            self.username.errors.append("Unknown username")
            print "Unknown username"
            return False

        if not user.check_password(self.password.data):
            self.password.errors.append('Invalid password')
            print "Wrong Password"
            return False

        self.user = user
        return True
		
@lm.user_loader
def load_user(user_id):
    '''The user_loader callback function used to reload the user object from the user id stored in the session'''
    return User.query.get(int(user_id))

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    '''Serves the login page if not logged in, or the main menu if logged in'''
    if g.user is not None and g.user.is_authenticated:
        username = g.user.username
        return redirect(url_for('manage_data', username=username))

    form = LoginForm()
    if form.validate_on_submit():
        session['remember_me'] = form.remember_me.data
        if 'remember_me' in session:
            remember_me = session['remember_me']
            session.pop('remember_me', None)
        login_user(form.user, remember=remember_me)
        username = form.username.data
        return redirect(url_for('manage_data', username=username))
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    '''Logout function requires login_required decorator which uses flask\'s login module'''
    logout_user()
    flash('You have logged out')
    return render_template('login.html', form=LoginForm())


@app.route('/profile')
@login_required
def profile():
    '''Serves the user profile page'''
    user = User.query.filter_by(username = g.user.username).first()
    return render_template('profile.html', user=user)

@app.route('/about')
def about():
    '''serves the about page'''
    return render_template('about.html')


