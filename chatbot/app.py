from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
from chatbot.recomodel import recomodel

app = Flask(__name__)
run_with_ngrok(app)

def make_reply(txt):

  rec = chatbot.recomodel()

  if "#" in txt:
      rec_book = rec.find_keyword_book(rec.get_genre_sim(), txt, 10)
  elif '-' in txt:
      rec_book = rec.find_rank_book(txt)
  else:
      print('엘스실행됨')
      rec_book = rec.find_sim_book(rec.get_genre_sim(), txt, 10)

  # colab 상에서는 그냥 클래스를 호출하지만 일반 IDE상에서는 recmodel.py를 따로 만들어서 모듈식으로 호출하면 좋음

  return rec_book.to_html()


@app.route('/', methods=['get'])
def main():
  m = ""
  res = ""
  return render_template('bbu-chat-room.html', txt=m, res=res)


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
