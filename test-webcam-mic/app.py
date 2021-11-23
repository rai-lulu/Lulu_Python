from flask import Flask,render_template,Response
import cv2

app = Flask (__name__,
            static_url_path='', 
            static_folder='./static',
            template_folder='./templates')

@app.route('/')
def index():
    return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)