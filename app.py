from flask import Flask, render_template,request,jsonify,redirect,url_for,session
from flask_mail import Mail, Message
import base64
import dlib
import scipy.misc
import numpy as np
import os
from werkzeug.utils import secure_filename
import sqlite3 as sql
import json
from random import randint
import cv2
import glob
import playsound
from gtts import gTTS

UPLOAD_FOLDER = './static/train'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

app.config.update(
    DEBUG=True,
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_SSL=True,
    MAIL_USERNAME = 'iamcool.rikshit@gmail.com',
    MAIL_PASSWORD = 'hemant0540'
)
mail = Mail(app)


def retrieveUserData():
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM userTbl")
    users = cur.fetchall()
    jsonarray = []
    for x in users:
        xasd =	{
            "name": x[1],
            "mobileNo": x[2],
            "emailID": x[3],
            "company":x[4],
            "photo_proof":x[5]
            }
        jsonarray.append(xasd)
    con.close()
    return jsonarray

def userData(userName):
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM userTbl where name= '%s' limit 1" %(userName))
    users = cur.fetchall()
    jsonarray = []  
    for x in users:
        xasd =	{
            "name": x[1],
            "mobileNo": x[2],
            "emailID": x[3],
            "company":x[4],
            "photo_proof":x[5]
            }
        jsonarray.append(xasd)
    con.close()
    return jsonarray

def insertUser(name,mobile,email,company,proof):
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("INSERT INTO userTbl (name, mobileNo, emailID, company,photo_proof) VALUES (?,?,?,?,?)", (name, mobile, email, company,proof))
    con.commit()
    con.close()

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
TOLERANCE = 0.4

def get_face_encodings(filename):
    image = scipy.misc.imread(filename)
    detected_faces = face_detector(image, 1) 
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)

def find_match(known_faces, names, face):
    matches = compare_face_encodings(known_faces, face)
    count = 0
    global dtr
    for match in matches:
        if match:
            dtr = names[count]
            return names[count]
        count += 1
        print(count)
    return 'Unknown'

#insert into record having timestamp
def inserttorecord(userid,dtr,name,name2):
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("Insert into Record (visitorid, visitorname, employeename, meetpurpose, TIMESTAMP) VALUES (?, ?, ?, ?, datetime('now','localtime'))", (userid,dtr,name,name2))
    con.commit()
    con.close()

# train data Api
@app.route('/trainData', methods=["POST","GET"])
def trainData():
    image = request.form.get("image")
    # print('imagessss',image)
    starter = image.find(',')
    image_data = image[starter + 1:]
    registerwala(image_data)
    convert_and_save(image_data)
    face_data = filter(lambda x: x.endswith('.npy'), os.listdir('faces/'))
    face_data = sorted(face_data)
    names = [x[:-4] for x in face_data]
    paths_to_facedata = ['faces/' + x for x in face_data]
    face_encodings = []
    for path in paths_to_facedata:
        face_encodings.append(np.load(path))   
    face_encodings_in_image = get_face_encodings('imageToSave.jpeg')
    if len(face_encodings_in_image) != 1:
        # print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        return jsonify(matchData= "it can only have one")
    match = find_match(face_encodings, names, face_encodings_in_image[0])
    data = userData(match)
    return jsonify(matchData=match,data=data)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#training data
@app.route('/trainingData', methods=["POST"])
def trainingData():
    image = request.form.get("image")
    name = request.form.get("name")
    emailID = request.form.get("emailID")
    company = request.form.get("company")
    mobileNo = request.form.get("mobileNo")
    IDproof = request.form.get("IDproof")
    insertUser(name,mobileNo,emailID,company,IDproof)
    starter = image.find(',')
    image_data = image[starter + 1:]
    trainData_and_save(name,image_data)

    print("nameeeeee",name)
    filenames = [img for img in glob.glob("static/train/{}.jpeg".format(name))]

    filenames.sort() # ADD THIS LINE

    images = []
    for img in filenames:
        n= cv2.imread(img)
        images.append(n)
        print("image",img)
        
    
    # final_face = get_face_encodings(img)
    np.save('faces/'+name, get_face_encodings(img)[0])
    # if request.method == 'POST':
    #     image_filenames = filter(lambda x: x.endswith('.jpeg'), os.listdir('static/train/'))
    #     image_filenames = sorted(image_filenames)
    #     no_of_images = len(image_filenames)
    #     names = [x[:-5] for x in image_filenames]
    #     paths = ['static/train/' + x for x in image_filenames]
    #     print(paths)
    #     for i in range(0,no_of_images):
    #         print(i)
    #         face_encodings_in_image = get_face_encodings(paths[i])
    #         print(face_encodings_in_image)
    #         print("names msadlkmf",names[i])
    #         if len(face_encodings_in_image) != 1:
    #             print("Please change image: " + paths[i] + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
    #             exit()
    #         np.save('faces/'+names[i],get_face_encodings(paths[i])[0])
    #         print("LAST print:",paths[i])
    return jsonify(matchData='successfully Train data')
    
@app.route('/visitorRecord')
def visitorRecord():
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("select vistorRecord.id ,vistorRecord.timeStamp, userTbl.name , Employee.employeeName ,Employee.employeeDesignation from ((vistorRecord inner join userTbl on vistorRecord.userID = userTbl.id) inner join Employee on vistorRecord.employeeID = Employee.id);")
    users = cur.fetchall()
    jsonarray = []
    for x in users:
        xasd =	{
            "id": x[0],
            "timeStamp": x[1],
            "VisitorName": x[2],
            "EmployeeName":x[3],
            "EmployeeDesignation":x[4]
            }
        jsonarray.append(xasd)
    con.close()
    print('asdasdasd',jsonarray)
    return jsonify(status='1',data=jsonarray)

@app.route('/')
def index(): 
    mytext = 'Welcome to R G I T'
  
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    myobj.save("welcomee.mp3")
    os.system("welcomee.mp3")
    return render_template('welcome.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/training')
def training():
    return render_template('Training.html')

@app.route('/otp', methods=['GET', 'POST'])
def otp():
    error = None
    if request.method == 'POST':
        print(nm)
        name = nm
        otp = request.form['otp']
        if name == otp:
            return redirect(url_for('meet'))
        else:
            error = 'invalid credential'
    return render_template('otp.html', error = error)

@app.route('/otp1')
def otp1():
    global nm
    print("namfor otp:", dtr)
    result = dtr
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("select * from userTbl where name = ? limit 1", (result,))
    resultt = cur.fetchall()
    for row in resultt:
        user = row[3]
    print("asbdjdhajskf", resultt)
    print("result:", user)
    msg = Message('Ricky', sender = 'iamcool.rikshit@gmail.com', recipients = [user])
    nm = str(randint(0, 9999))
    print(nm)
    msg.body = "Hello your otp is {}".format(nm)
    mail.send(msg)
    print("nameotp:", nm)
    return redirect(url_for('otp'))

@app.route('/meet',methods = ['POST', 'GET'])
def meet():
    result = dtr
    print('resultmeetwala', result)
    con = sql.connect("Visito.db")
    cur = con.cursor()
    cur.execute("select * from Employee")
    users = cur.fetchall()
    jsonarray = []
    for x in users:
        xasd =	{
            "id": x[0],
            "employeeName": x[1],
            # "employeeEmail": x[2],
            # "employeePhone":x[3],
            # "employeeDesignation":x[4]
            }
        jsonarray.append(xasd)
    con.close()
    print('jsonarray:',jsonarray)
    json_list = json.dumps(jsonarray)
    print("jsondump", json_list)
    # return jsonify(status='1',data=jsonarray)
    con = sql.connect("Visito.db")
    cur = con.cursor()
    cur.execute("select * from Purpose")
    users = cur.fetchall()
    jsonarrayy = []
    for x in users:
        xasdd =	{
            "purpose": x[0]
            }
        jsonarrayy.append(xasdd)
    con.close()
    print('jsonarray:',jsonarrayy)
    json_list2 = json.dumps(jsonarrayy)
    print("jsondump", json_list2)

    return render_template('meet.html', result=result, json_list=json_list, json_list2=json_list2)

@app.route('/cheethhi', methods = ['POST', 'GET'])
def cheethhi():
    global username
    name = request.form.get("name")
    id = request.form.get("id")
    name2 = request.form.get("name2")
    print("purpose:", name2)
    print("cheethiname", name)
    print("cheethiID", id)
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("select * from Employee where employeeName = ? limit 1", (name,))
    user = cur.fetchall()
    for row in user:
        user = row[2]
        username = row[1]
    print("cheethiemail", user)
    msg = Message('Ricky', sender = 'iamcool.rikshit@gmail.com', recipients = [user])
    # msg.body = "Hello {} is coming to meet you".format(dtr)
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("select * from userTbl where name = ? limit 1", (dtr,))
    usr = cur.fetchall()
    for row in usr:
        userid = row[0]
        usr = row[4]
        usr2 = row[3]
        userno = row[2]
    msg.html = render_template('emailotp.html', username=username, dtr=dtr, usr=usr, usr2=usr2, name2=name2, userno=userno)
    with app.open_resource("imageToSave.jpeg") as fp:
        msg.attach("imageToSave.jpeg", "imageToSave/jpeg", fp.read())
    mail.send(msg)
    inserttorecord(userid,dtr,name,name2)
    return jsonify(matchData=username)

@app.route('/emailotp')
def emailotp():
    return 'ok'

@app.route('/conformation',methods = ['POST','GET'])
def conformation():
    name = username
    result = dtr
    mytext = 'Just Have a seat {}  {} has been informed and will joining you shortly'.format(result, name)
  
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    myobj.save("welcomee.mp3")
    os.system("welcomee.mp3")
    return render_template('conformation.html', name=name, result=result)

@app.route('/register',methods = ['POST', 'GET'])
def register():
    print("register wala name:", nameregister)
    result = nameregister
    resultimage = imageurl2
    print("result name:", result)
    return render_template('register.html', result=result, resultimage=resultimage)

@app.route('/VisitorDetail',methods = ['POST', 'GET'])
def VisitorDetail():
    # if request.method == 'POST':
    print("register wala name:", nameregister)
    result2 = nameregister
    print("result name:", result2)
    print("nam:", dtr)
    resultt = dtr
    con = sql.connect("visito.db")
    cur = con.cursor()
    cur.execute("select * from userTbl where name = ? limit 1", (resultt,))
    print("asd", resultt)
    result = cur.fetchall()
    print("asd2", result)
    mytext = 'Welcome Back {}'.format(resultt)
  
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    myobj.save("welcomee.mp3")
    os.system("welcomee.mp3")
    return render_template('VisitorDetail.html', result=result, result2=result2)
    # result = request.form.get("id")
    # print("resulttttt", result)
    # return render_template("VisitorDetail.html", result=result)

# retrieve data
@app.route('/userData')
def retrieveUsers():
    data = retrieveUserData()
    return json.dumps(data)

def convert_and_save(b64_string):
    # print('test',b64_string)
    with open("imageToSave.jpeg", "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))

def trainData_and_save(name,b64_string):
    with open("static/train/"+name+".jpeg", "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))

def registerwala(img_url):
    # print("imgsakjdk", img_url)
    global nameregister
    # print('ijs', img_url)
    print("list of image",len(os.listdir("Static/image/")))
    if (len(os.listdir("static/image/")) == 10):
        deleteregisterwala()
        print("delete successfully")
    nameregister = str(randint(0, 10000))
    print("nameeeeeeee", nameregister)
    second(nameregister, img_url)
    return jsonify(status='1')
    # with open("static/image/imagesave.jpeg", "wb") as fh:
    #     fh.write(base64.decodebytes(img_url.encode()))
        
def second(name, img_url):
    global imageurl2
    imageurl2 = img_url
    with open("static/image/"+name+".jpeg", "wb") as fh:
        fh.write(base64.decodebytes(imageurl2.encode()))


@app.route('/deleteregisterwala')
def deleteregisterwala():
    filelist = [f for f in os.listdir("static/image/") if f.endswith(".jpeg")]
    for f in filelist:
        os.remove(os.path.join("static/image/", f))

# train data Api
@app.route('/slingshoot', methods=["POST"])
def slingApi():
    print("list of SlingShot",len(os.listdir("Sling/")))
    if (len(os.listdir("Sling/")) == 10):
        deleteSlingData()
        print("delete successfully")
    image = request.form.get("image")
    print("images",image)
    name = str(randint(0, 10000))
    print('nameeeee',name)
    slingData(name,image)
    return jsonify(status='1')

def slingData(name,b64_string):
    with open("Sling/"+name+".jpeg", "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))

@app.route('/Dslingshoot')
def deleteSlingData():
    filelist = [ f for f in os.listdir("Sling/") if f.endswith(".jpeg") ]
    for f in filelist:
        os.remove(os.path.join("Sling/", f))
    # return jsonify(status='Remove Successfully')

@app.route('/music')
def music():
    playsound.playsound('music.mp3', False)
    print('asdasdasd')
    # winsound.PlaySound("static/music/music.mp3", winsound.SND_NOSTOP )
    # winsound.PlaySound(None, winsound.SND_ASYNC)
    return 'ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

# select vistorRecord.id ,vistorRecord.timeStamp, userTbl.name , Employee.employeeName from ((vistorRecord inner join userTbl on vistorRecord.userID = userTbl.id)
# inner join Employee on vistorRecord.employeeID = Employee.id);

# save Session
# session['user'] = 'save data'

# get session
# if 'user' in session:
#     session['user']

# drop session
# session.pop('user',None)