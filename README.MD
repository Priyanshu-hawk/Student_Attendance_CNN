# Student_Attendance_CNN

## How to use

## Create a virtual environment and install all the requirements from requirements.txt
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt -vvv
```

### Put all you images in the folder "img" for training.

### Run the following command start marking attendece.

```
python3 atten_proj.py
```

## Above command will create a file "attendance_d_m_y.csv" which will contain the name of the student and the time of attendance.

## And also it will create a 'stud.pkl' file to store encodings and names.