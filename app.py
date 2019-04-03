import os,cv2
from flask import Flask, render_template, request,jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ocr', methods=['POST','GET'])
def upload_file():
    if request.method == "POST":
        file = request.files['image']

        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
        file.save(f)
        print(file.filename)

        image = cv2.imread(UPLOAD_FOLDER+"/"+file.filename)
        #
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, image)
        print(filename)
        
        CATEGORIES = ["NORMAL", "PNEUMONIA"] # so that our model classifies in these labels only


        def prepare(filepath): # we have to adjust our input image according to our training data set dimensions
            IMG_SIZE = 150  # 50 in txt-based
            img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        model = tf.keras.models.load_model("PneumoniaModel.h5") # this is tensorflow serving importing an keras deep leaerning save module

        prediction = model.predict([prepare(filename)])
        os.remove(filename)
        d=float(prediction[0][0])
        text= (CATEGORIES[int(prediction[0][0])])
        text= text+"\t"+ str(d)+'\n'+"Filename: \t"+str(file.filename)
        print("Classification :\n",text)
        return jsonify({"text" : text})
        os.remove(UPLOAD_FOLDER+"/"+file.filename)
        
if __name__ == '__main__':
	app.run(debug=True) #host="0.0.0.0"
#app.run(threaded=True,debug=True)


