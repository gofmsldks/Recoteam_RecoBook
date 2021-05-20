from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def make_reply(txt):
  return 'Hello.'


@app.route('/', methods=['get'])
def main():
  m = ""
  res = ""
  return render_template('chatbot.html', txt=m, res=res)


@app.route('/chatbot', methods=['get'])
def chatbot():
  m = request.args.get('m')
  txt = request.args.get('txt')
  res = ""

  if m == "say" :
    res = make_reply(txt)

  # print(jsonify(result=res))
  return  res

if __name__ == '__main__':
    app.run()
