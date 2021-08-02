from flask import *
import summ
app=Flask(__name__)
@app.route('/')
def first_page():
    return render_template("webpage.html")
@app.route('/main',methods=['POST'])
def main():
    if request.method=='POST':
        file=request.files['pdf']
        file.save(file.filename)
        end_dict=summ.summ(file.filename)
        return render_template("main.html",end_dict=end_dict)
    else:
        return render_template("webpage.html")
if __name__=='__main__':
    app.run(debug=True)
