import music21 as m21
import numpy as np
import math
import glob
import itertools
import pandas as pd
import json
import csv
import collections
import sys

path_str = "" # 絶対パスを指定したい人はココを変更してください（このプログラム一式があるディレクトリを指定）
filename = "data/sample.mxl" # 自分が持っているmxlファイルで編曲したい方は，ここを変更してください 

### 難易度を選択してください
kind_of_difficulty = ["easy", "hard"] # ここは変更しないでください
primo_difficulty = "hard" ## Primoの難易度を，"easy" or "hard"　で選択
secondo_difficulty = "hard" ## Secondoの難易度を，"easy" or "hard"　で選択
your_difficulty =[kind_of_difficulty.index(primo_difficulty), kind_of_difficulty.index(secondo_difficulty)] # ここは変更しないでください

### 開始小節番号と小節数を変更してください
start_bar = 0  # 開始小節番号（0スタート）
bar_len = 20  # 小節数

### これ以降はプログラムをいじる部分はありません

s = m21.converter.parse(filename)


def flatten(l):
  for el in l:
    if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
      yield from flatten(el)
    else:
      yield el

def getMelody(filename, part_id):  # dicを返す
  pitchs = []    # 音名
  note_num = []  # ノートナンバー
  lengths = []   # 音価
  articulations = []
  tie = []
  bar_id = []
  beat_id = []
  for n in filename.parts[part_id].flat.notesAndRests:
    temp_p = []
    temp_notenum = []
    if isinstance(n, m21.note.Rest):  # 休符
      temp_p.append(n.name)
      temp_notenum.append(-1)
    elif isinstance(n, m21.note.Note):  # 単音
      temp_p.append(n.name)
      temp_notenum.append(int(m21.pitch.Pitch(n.nameWithOctave).ps))
    elif isinstance(n, m21.chord.Chord):  # コード
      for chord_note in n.pitches:
        temp_p.append(chord_note.name)
        temp_notenum.append(int(m21.pitch.Pitch(chord_note.nameWithOctave).ps))
    else:
      continue
    pitchs.append(temp_p)
    note_num.append(temp_notenum)
    if isinstance(n, m21.note.Rest):  # 休符
      lengths.append(-(n.duration.quarterLength))
    else:
      lengths.append(n.duration.quarterLength)
    articulations.append(n.articulations)
    if isinstance(n.tie, m21.tie.Tie):  # タイ
      tie.append(n.tie.type)
    else:
      tie.append('none')
    bar_id.append(n.measureNumber-1)
    beat_id.append([beat for beat in range(int(math.floor(n.offset)), int(math.ceil(n.offset+abs(n.duration.quarterLength))))])

  data = dict()                            # 辞書を生成
  data['pitch'] = pitchs                   # 2 str
  data['note_num'] = note_num              # 2 int（休符は-1）
  data['length'] = lengths                 # 1 float（休符はその長さを負の数で）
  data['articulation'] = articulations     # 1 m21 ?
  data['tie'] = tie                        # 1 str（無い場合は'none'）
  data['bar_id'] = bar_id                  # int（0スタート）
  data['beat_id'] = beat_id                # 2 int（何拍めに鳴っている音か）
  return data  # 辞書型

def getBarInfo(filename, start_bar, bar_interval):
  bar_dic = dict()
  for bar_id in range(start_bar, start_bar +bar_interval):
    tempo = False
    key_signature = False
    time_signature = False
    for element in filename.parts[0].measure(bar_id +1).elements:  # ()の中は小節数（1スタート）
      if element.__class__.__name__ == "MetronomeMark":
        tempo = element._number
      elif element.__class__.__name__ == "KeySignature":
        key_signature = element._sharps
      elif element.__class__.__name__ == "TimeSignature":
        time_signature = str(element.displaySequence)[1:-1]
    bar_dic[bar_id] = [tempo, key_signature, time_signature]
  return bar_dic

def getDataX(nn, l):
  data = []

  # フレーズの先頭に音がある
  hasNote = True
  if l[0] < 0:
    hasNote = False

  # 「休符」は除外する
  new_nn = []  # 2次元配列になる
  new_l = []   # 1次元配列
  for n in nn:
    if sum(n) >= 0:
      new_nn.append(n)
  for length in l:
    if length >= 0:
      new_l.append(length)

  if len(new_l) != 0: 
    new_nn_all = list(flatten(new_nn))  # 1次元配列にする

    # ノートナンバーの平均
    nn_mean = np.average(new_nn_all)

    # 最高音と最低音の差
    nn_high = int(max(new_nn_all))
    nn_low = int(min(new_nn_all))
    nn_dif = nn_high - nn_low
    
    # 音符の数
    l_num = len(new_l)

    # NEW!!
    # 2分音符以上の数
    num_of_halfNote = 0
    for n_len in new_l:
      if n_len >= 2.0:
        num_of_halfNote += 1

    # 最頻の音価の割合
    c = collections.Counter(new_l)
    frequent_note_value = c.most_common()[0][0]
    frequent_note_value_n = new_l.count(frequent_note_value)
    note_value_ratio = frequent_note_value_n / len(new_l)

    # 隣あう音高の差（の2乗平均）
    nn_diff_list = []
    for i in range(len(new_nn)):
      if i == 0: continue
      nn_diff_list.append(max(new_nn[i]) - max(new_nn[i-1]))
    if len(nn_diff_list) > 0:
      nn_diff_mean = sum(np.power(nn_diff_list, 2)) / len(nn_diff_list)
    else:
      nn_diff_mean = 0.0
  else:
    nn_mean = -1
    nn_dif = -1
    l_num = -1
    num_of_halfNote = -1
    note_value_ratio = -1
    nn_diff_mean = -1
  
  data.append(hasNote)
  data.append(nn_mean)
  data.append(nn_dif)
  data.append(l_num)
  data.append(num_of_halfNote)
  data.append(note_value_ratio)
  data.append(nn_diff_mean)

  return data

### JSONデータを取得(ファイル内全て)
json_files = glob.glob('%s/input_json/*.json*' % path_str)
print(json_files)
inputX = []
inputY = []
for jf in json_files:
  with open(jf, mode='rt', encoding='utf-8') as f:
    # 辞書オブジェクト(dictionary)を取得
    data = json.load(f)
    inputX += data['inputX']
    inputY += data['inputY']

### fitする
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
inputX_std = scaler.fit_transform(inputX)
mean, scale = scaler.mean_, scaler.scale_
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(inputX_std, inputY)

def getOutputY(divided_file):
  outputX = []
  for part_id in range(len(divided_file.parts)):
    notes_info_dic = getMelody(divided_file, part_id)
    note_nums, lengths = notes_info_dic['note_num'], notes_info_dic['length']
    outputX.append(getDataX(note_nums, lengths))
  outputX_std = scaler.transform(outputX)
  outputY = neigh.predict(outputX_std)
  return outputY

def getClass(divided_file, n_melody):  # 同じ役割と判断したものをまとめる
  output = getOutputY(divided_file)
  result = [[] for j in range(n_melody)]
  # 同クラスの番号を配列に格納する
  for i in range(len(output)):
    for j in range(n_melody):
      if output[i] == j:
        result[j].append(int(i))
  return result

def getMinLen(filename):  # 音価と発音時刻のみを見て，全く同じならグループとして扱う
  length_all = []
  for part_id in range(len(filename.parts)):  # パートを回す
  # for part_id in range(1, 2):  # パートを回す
    notes_info_dic = getMelody(filename, part_id)
    lengths = notes_info_dic['length']
    length_all.append(lengths)
  min_length = float(min([l for l in np.abs(list(flatten(length_all))) if l != 0.0]))
  return min_length

def sonicRichness(bar, l, min_l):  # 発音時刻，発音持続パターン
  cnt = 1
  if l >= 0:
    v = 2
    bar.append(int(v))
    v -= 1
    while min_l * cnt < abs(l):
      bar.append(int(v))
      cnt += 1
  else:
    v = 0
    bar.append(int(v))
    while min_l * cnt < abs(l):
      bar.append(int(v))
      cnt += 1

def assVector(filename):  # ベクトルを付与する
  min_length = getMinLen(filename)
  ans = []
  for part_id in range(len(filename.parts)):  # パートを回す
    notes_info_dic = getMelody(filename, part_id)
    lengths = notes_info_dic['length']
    var_SR = []  # 発音持続パターン
    for i in range(len(lengths)):  # 小節の中身を回す
      l = lengths[i]
      sonicRichness(var_SR, l, min_length)
    sr = ''.join([str(n) for n in var_SR])
    ans.append(int(sr))
  return ans

def getPartGroups(divided_file, melodies_number):  # グループ分け→その中で全く同じフレーズ（発音時刻，音価）をまとめて，最終的に使用するパート群全てを2次元配列で返す
  ans = []
  for n in melodies_number:
    isRest = (n == 4)
    if not isRest:
      new_s = m21.stream.Score()
      for part_id in outputY[n]:
        new_s.append(divided_file.parts[part_id])
      vec = assVector(new_s)
      c = collections.Counter(vec)
      frequent_vec = c.most_common()[0][0]

      group_ans = []
      for i in range(len(outputY[n])):
        # vecの中で全く同じものをグループ化する
        if frequent_vec == vec[i]:
          group_ans.append(outputY[n][i])
      ans.append(group_ans)
    else:  # 休符の時
      ans.append([-1])
  return ans

def decideMelodies(n_hands, difficulty):
  melodies_number = []
  for i in range(len(outputY)):
    if (len(outputY[i]) > 0 and i < 4):
      melodies_number.append(i)
  if len(melodies_number) == 4:
  # 全部あるとき
    melodeis_numbers = [[[[0, 4, 4, 3], [0, 4, 3, 3], [0, 0, 4, 3]],
                         [[0, 0, 2, 3], [2, 2, 0, 3], [0, 0, 1, 3], [1, 1, 0, 3]]],
                        [[[0, 2, 3, 3], [0, 1, 4, 3]],
                         [[0, 2, 1, 3], [1, 2, 0, 3], [0, 1, 2, 3]]]]
  elif len(melodies_number) == 2:
    if set([2, 3]) == set(melodies_number):
    # 0と1が無い時
      melodeis_numbers = [[[[2, 2, 4, 3], [4, 2, 4, 3]],
                          [[2, 2, 4, 3], [4, 2, 4, 3]]],
                         [[[2, 2, 4, 3], [4, 2, 4, 3]],
                          [[2, 2, 4, 3], [4, 2, 4, 3]]]]
    elif set([1, 3]) == set(melodies_number):
    # 0と2が無い時
      melodeis_numbers = [[[[4, 1, 4, 3]],
                           [[4, 1, 4, 3]]],
                          [[[4, 1, 4, 3]],
                           [[4, 1, 4, 3]]]]
    elif set([1, 2]) == set(melodies_number):
    # 0と3が無い時
      melodeis_numbers = [[[[1, 4, 2, 4], [4, 1, 4, 2]],
                           [[1, 4, 2, 4], [4, 1, 4, 2]]],
                          [[[1, 4, 2, 4], [4, 1, 4, 2]],
                           [[1, 4, 2, 4], [4, 1, 4, 2]]]]
    elif set([0, 3]) == set(melodies_number):
    # 1と2が無い時
      melodeis_numbers = [[[[0, 4, 4, 3], [0, 4, 3, 3], [0, 0, 4, 3]],
                           [[0, 4, 4, 3], [0, 4, 3, 3], [0, 0, 4, 3]]],
                          [[[0, 4, 4, 3], [0, 4, 3, 3], [0, 0, 4, 3]],
                           [[0, 4, 4, 3], [0, 4, 3, 3], [0, 0, 4, 3], [0, 3, 0, 3]]]]
    elif set([0, 2]) == set(melodies_number):
    # 1と3が無い時
      melodeis_numbers = [[[[0, 4, 2, 4], [0, 4, 2, 4], [0, 4, 2, 2], [4, 2, 0, 0]],
                           [[0, 4, 0, 2], [4, 2, 0, 2]]],
                          [[[0, 2, 0, 4]],
                           [[0, 2, 0, 2]]]]
    elif set([0, 1]) == set(melodies_number):
    # 2と3が無い時
      melodeis_numbers = [[[[0, 4, 1, 1], [0, 4, 1, 4]],
                           [[0, 4, 1, 1], [0, 4, 1, 4]]],
                          [[[0, 4, 1, 1], [0, 4, 1, 4]],
                           [[0, 4, 1, 1], [0, 4, 1, 4]]]]
  elif len(melodies_number) == 3:
    if 0 not in melodies_number:
    # 0が無い時
      melodeis_numbers = [[[[2, 2, 4, 3], [1, 4, 2, 4]],
                          [[1, 1, 2, 3]]],
                         [[[1, 2, 3, 3]],
                          [[1, 1, 2, 3], [1, 2, 3, 3]]]]
    elif 1 not in melodies_number:
    # 1が無い時
      melodeis_numbers = [[[[2, 2, 4, 3], [0, 4, 3, 3], [0, 4, 2, 2]],
                          [[0, 4, 2, 3], [4, 2, 0, 3], [0, 0, 2, 3], [2, 2, 0, 3]]],
                         [[[0, 2, 4, 3], [0, 2, 3, 3]],
                          [[0, 2, 0, 3]]]]
    elif 2 not in melodies_number:
    # 2が無い時
      melodeis_numbers = [[[[4, 1, 4, 3], [0, 4, 3, 3], [0, 4, 1, 1]],
                          [[0, 4, 1, 3], [0, 0, 1, 3], [1, 1, 0, 3]]],
                         [[[0, 1, 4, 3]],
                          [[0, 4, 1, 3], [0, 0, 1, 3], [1, 1, 0, 3], [0, 1, 4, 3]]]]
    elif 3 not in melodies_number:
    # 3が無い時
      melodeis_numbers = [[[[0, 4, 1, 4], [0, 4, 2, 4], [1, 4, 2, 4]],
                          [[0, 0, 1, 2]]],
                         [[[0, 2, 1, 4], [0, 1, 2, 2]],
                          [[0, 2, 0, 1]]]]
  elif len(melodies_number) == 1:
    if 0 in melodies_number:
    # 0しか無い時
      melodeis_numbers = [[[[0, 4, 4, 4]], [[0, 4, 4, 4]]], [[[0, 4, 4, 4]], [[0, 4, 4, 4]]]]
    elif 1 in melodies_number:
    # 1しか無い時
      melodeis_numbers = [[[[1, 4, 1, 4]], [[1, 4, 1, 4]]], [[[1, 4, 1, 4]], [[1, 4, 1, 4]]]]
    elif 2 in melodies_number:
    # 2しか無い時
      melodeis_numbers = [[[[2, 4, 2, 4]], [[2, 4, 2, 4]]], [[[2, 4, 2, 4]], [[2, 4, 2, 4]]]]
    elif 3 in melodies_number:
    # 3しか無い時
      melodeis_numbers = [[[[4, 4, 4, 3]], [[4, 4, 4, 3]]], [[[4, 4, 4, 3]], [[4, 4, 4, 3]]]]
  else:
  # 全部無い時
    melodeis_numbers = [[[[4, 4, 4, 4]], [[4, 4, 4, 4]]], [[[4, 4, 4, 4]], [[4, 4, 4, 4]]]]
  
  kind_of_melodies = ["主旋律", "副旋律", "和声　", "ベース", "休符　", "その他"]
  candidate_melodeis = melodeis_numbers[difficulty[0]][difficulty[1]]
  
  for i in range(len(candidate_melodeis)):
    print("%2d: " % i, end='')
    for mlody_i in candidate_melodeis[i]:
      print("%s　" % kind_of_melodies[mlody_i], end='')
    print()
  print("使用したい番号を半角で入力>>> ", end='')
  input_id = int(input())
  if not(input_id in list(range(len(candidate_melodeis)))):
    print("入力が正しくありません．処理を終了します．")
    sys.exit()
  else:
    print("選択した旋律は，")
    print("%d: " % input_id, end='')
    for mlody_i in candidate_melodeis[input_id]:
      print("%s　" % kind_of_melodies[mlody_i], end='')
    print("です．")
    print()
  ans = candidate_melodeis[input_id]
  return ans

def getRest(start_bar, interval, time_signatures):
  note_num, length, bar_id = [], [], []
  for i in range(interval):
    bar_beat_n = int(time_signatures[i][0]) * (4/int(time_signatures[i][-1]))  # 1小節の拍数（4分音符 = 1.0）
    note_num.append([-1])
    length.append(-bar_beat_n)
  for bar in range(start_bar, start_bar + interval):
    bar_id.append(bar)
  return note_num, length, bar_id

def getChord(in_note_num):  # 1回に弾く音符（和音のときもある） in_note_num: [[], [], [], ...]
  note_num = list(flatten(in_note_num))
  ans_notes = [-1]
  if max(note_num) >= 0:
    notes = []
    temp_notes = [n for n in note_num if n >= 0]
    temp_notes = list(set(temp_notes))
    ## 1オクターブ未満に収まるようにする
    for n in temp_notes:
      while max(temp_notes) - n > 12 -1:  # 1オクターブに収まるまで繰り返す
        n += 12
      notes.append(n)
    notes = sorted(list(set(notes)), reverse=True)
    ## n和音以上の時に3和音にする
    if len(notes) <= 3:  # 何音以内か
      ans_notes = notes
    else:
      cur_note = notes[0]
      ans_notes = [cur_note]
      for nn in notes:
        if 3 <= cur_note - nn <= 6:
          ans_notes.append(nn)
          cur_note = nn
        if len(ans_notes) >= 3: break
  return ans_notes

def createOnePhrase(all_note_n):  # 1フレーズ（1パートのみ）のノートナンバーを返す
  fhrase_nn = []
  for i in range(len(all_note_n[0])):
    chord_nn = []
    for j in range(len(all_note_n)):
      chord_nn.append(all_note_n[j][i])
    fhrase_nn.append(getChord(chord_nn))
  return fhrase_nn

def getPhraseNotenum(filename, melody):  # 1フレーズ（全パート）のノートナンバーを返す
  all_note_n = []
  for part_id in melody:
    temp_dic = getMelody(filename, part_id)
    all_note_n.append(temp_dic['note_num'])
  fhrase_nn = createOnePhrase(all_note_n)
  return fhrase_nn

def getPhraseWithoutNotenum(filename, melody):  # 1フレーズ（全パート）のノートナンバー以外を返す
  temp_dic = getMelody(filename, melody[0])
  all_length = temp_dic['length']
  all_artic = temp_dic['articulation']
  all_tie = temp_dic['tie']
  all_beat_id = temp_dic['beat_id']
  all_bar_id = temp_dic['bar_id']
  return all_length, all_artic, all_tie, all_beat_id, all_bar_id

def getPlausibilityJson():
  filname_str = "FourHands_plausibility"
  pitch_mean = ''
  pitch_var = ''
  with open('%s/%s.json' % (path_str, filname_str), mode='rt', encoding='utf-8') as f:
    # 辞書オブジェクト(dictionary)を取得
    data = json.load(f)
    pitch_mean = data['mean']
    pitch_var = data['var']
  return pitch_mean, pitch_var

def getScorePitchRange(notes, beat_id):  # 連弾全体のフレーズのスコアを返す
  mean_corpus, var_corpus = getPlausibilityJson()
  flat_notes = []  # 1パートにつき1次元，休符を抜く, [[], [], [], []]
  flat_beat_ids = []  # 1パートにつき1次元，休符を抜く, [[[], [], ...], [[], [], ...], [[], [], ...], [[], [], ...]]，何拍目に鳴っているかだけを見ている
  for part_id in range(len(notes)):
    temp_notes = []
    temp_beat_ids = []
    for i in range(len(notes[part_id])):
      for j in range(len(notes[part_id][i])):
        if notes[part_id][i][j] >= 0:
          temp_notes.append(notes[part_id][i][j])
          temp_beat_ids.append(beat_id[part_id][i])
    flat_notes.append(temp_notes)
    flat_beat_ids.append(temp_beat_ids)
  hasNote = [bool(len(flat_n)) for flat_n in flat_notes]

  ## mean
  pitch_mean = []  # [, , , ]
  for part_id in range(len(notes)):
    if hasNote[part_id]:
      pitch_mean.append(np.mean(flat_notes[part_id]))
    else:
      pitch_mean.append(-1)
  
  mean_diff = []  # 平均とどれくらい離れているか
  for part_id in range(len(notes)):
    if(hasNote[part_id]):
      mean_diff.append(abs(mean_corpus[part_id] - pitch_mean[part_id]))
    else:  # 休みのとき
      mean_diff.append(0.0)

  ## ovarlap
  pitch_diff = 2  # 音域がどれくらい離れていれば重なっていないとするか
  cnt = [0 for _ in range(len(notes))]  # 重なっている音符数

  beat_ids_n = list(flatten(flat_beat_ids)) # 少なくともどこかには旋律がある前提
  for now_id in range(min(beat_ids_n), max(beat_ids_n)+1):
    pitch_range = []  # [[, ], [, ], [, ], [, ]]
    hasRest = False
    temp_nns = []
    for part_id in range(len(notes)):
      temp_n = []
      for i in range(len(flat_notes[part_id])):
        if now_id in flat_beat_ids[part_id][i]:
          temp_n.append(flat_notes[part_id][i])
      if temp_n:
        pitch_range.append([max(temp_n), min(temp_n)])
      else:
        if not(0 < part_id < len(notes)-1):  # 端っこの時
          pitch_range.append([-999, 999])
        else:
          hasRest = True
          pitch_range.append([])
      temp_nns.append(temp_n)
    if hasRest:
      for part_id in range(len(notes)):
        if not(len(pitch_range[part_id])) and 0 < part_id < len(notes)-1:
          temp_max, temp_min = '', ''
          if len(pitch_range[part_id+1]):
            temp_max = pitch_range[part_id+1][0]
          else:
            temp_max = pitch_range[part_id+2][0]
          if len(pitch_range[part_id-1]):
            temp_min = pitch_range[part_id-1][1]
          else:
            temp_min = pitch_range[part_id-2][1]
          pitch_range[part_id] = [temp_max, temp_min]
    
    for part_id in range(len(notes)):
      if part_id > 0:
        for i in range(len(temp_nns[part_id])):
          if temp_nns[part_id][i] > pitch_range[part_id -1][1] - pitch_diff:
            cnt[part_id] += 1
            break
      if part_id < len(notes)-1:
        for i in range(len(temp_nns[part_id])):
          if temp_nns[part_id][i] < pitch_range[part_id +1][0] + pitch_diff:
            cnt[part_id] += 1
            break

  w = [0.6, 0.4]  # それぞれどれを重視したいかという重み
  score = []
  for part_id in range(len(notes)):
    norm_mean_diff = math.exp(-(mean_diff[part_id]**2 / var_corpus[part_id]))
    beat_n_all = max(beat_ids_n) +1 -min(beat_ids_n)
    if beat_n_all != 0:
      norm_cnt = (beat_n_all - cnt[part_id]) / beat_n_all  # 拍数で割る
    else:
      print("拍数が0だけど，合ってる？")
      norm_cnt = 1.0
    part_score = w[0] * norm_mean_diff + w[1] * norm_cnt
    score.append(part_score)
  return sum(score)

def moveOctave(note, n_octave):  # 1つのパートのオクターブを移動する
  one_octave = 12
  phrase = []
  for nn in note:
    if max(nn) >= 0:
      temp = []
      for n in nn:
        if 0 <= n+n_octave*one_octave <= 128:
          temp.append(n+n_octave*one_octave)
        else:
          temp.append(-1)
      phrase.append(temp)
    else:
      phrase.append([n for n in nn])
  return phrase

def adjustPitchRange(notes, beat_id):  # どのオクターブの組み合わせが一番良いかを決定し，連弾のフレーズのノートナンバーを返す
  octaves = []  # 5 * 4
  for note in notes:
    octaves_melody = []
    for n_octave in range(2, -3, -1):  # 2, 1, 0, -1, -2
      octaves_melody.append(moveOctave(note, n_octave))
    octaves.append(octaves_melody)
  score = []
  comb = itertools.product(*octaves)  # 全通り補完する
  for oct in comb:
    oct_n = list(oct)
    flat_oct_n = [n for n in list(flatten(oct_n)) if n != -1]
    if(0 <= min(flat_oct_n) <= 128):  # 休符のみを削除する
      temp_score = getScorePitchRange(list(oct), beat_id)
      score.append(temp_score)
    else:
      score.append(0.0)
  max_id = score.index(max(score))
  comb = itertools.product(*octaves)  # なぜか1度使用すると値が全て消えるので，もう一度
  return list(list(comb)[max_id])

def getEnharmonic(notes, key_signature):
  pitches = []
  enharmonic = [["C#", "D-"], ["D#", "E-"], ["F#", "G-"], ["G#", "A-"], ["A#", "B-"]]
  for i in range(len(notes)):
    p = str(m21.pitch.Pitch(midi = notes[i]).nameWithOctave)
    if len(p) > 1:
      if key_signature <= 0:
        if p[1] =='#':
          for enh in enharmonic:
            if p[0:-1] in enh: p = str(enh[1]) + str(p[-1])
      else:
        if p[1] == '-':
          for enh in enharmonic:
            if p[0:-1] in enh: p = str(enh[0]) + str(p[-1])
    pitches.append(p)
  return pitches

def inMeas(stream, fhrase_nn, length, artic, tie, bar_id, bar_info_dic):  ### 基本的にm21型にしてmeasに入れるだけをする
  global global_keySignature
  pre_bar_id = bar_id[0]
  meas = m21.stream.Measure()
  ## ココでnote型にする
  ## もしテンポ，調号，拍子が変わっていたら，変更する
  if bar_info_dic[pre_bar_id][0]:
    tempo = bar_info_dic[pre_bar_id][0]
    stream.append(m21.tempo.MetronomeMark(number=tempo))
  if bar_info_dic[pre_bar_id][1]:
    key_signature = bar_info_dic[pre_bar_id][1]
    stream.append(m21.key.KeySignature(key_signature))
    global_keySignature = key_signature
  if bar_info_dic[pre_bar_id][2]:
    time_signature = bar_info_dic[pre_bar_id][2]
    stream.append(m21.meter.TimeSignature(time_signature))
  for i in range(len(length)):
    if pre_bar_id < bar_id[i] and 0 < i: # 小節が1つ進んだら，前のフレーズを1小節に入れ，値を初期化する
      stream.append(meas)
      stream.makeMeasures(inPlace=True)
      pre_bar_id = bar_id[i]
      meas = m21.stream.Measure()
      ## もしテンポ，調号，拍子が変わっていたら，変更する
      if bar_info_dic[bar_id[i]][0]:
        tempo = bar_info_dic[bar_id[i]][0]
        stream.append(m21.tempo.MetronomeMark(number=tempo))
      if bar_info_dic[bar_id[i]][1]:
        key_signature = bar_info_dic[bar_id[i]][1]
        stream.append(m21.key.KeySignature(key_signature))
        global_keySignature = key_signature
      if bar_info_dic[bar_id[i]][2]:
        time_signature = bar_info_dic[bar_id[i]][2]
        stream.append(m21.meter.TimeSignature(time_signature))
    notes = fhrase_nn[i]
    quarter_length = float(abs(length[i]))
    isRest = False
    n = ''
    if max(notes) < 0:
      n = m21.note.Rest(quarterLength = quarter_length)
      isRest = True

    if not isRest:
      pitches = getEnharmonic(notes, global_keySignature)
      p = [m21.pitch.Pitch(cur_p) for cur_p in pitches]
      if len(p) == 1:
        n = m21.note.Note(p[0], quarterLength = quarter_length)
        if quarter_length == 0.0:
          n = n.getGrace()
      else:
        n = m21.chord.Chord(p, quarterLength = quarter_length)
        if quarter_length == 0.0:
          n = n.getGrace()
      n.articulations = artic[i]
      if tie[i] != 'none':
        n.tie = m21.tie.Tie(tie[i])
    
    meas.append(n)
  stream.append(meas)
  stream.makeMeasures(inPlace=True)

print("● 編曲を開始します．", end="\n\n")
### 編曲前の準備，おまじない的なやつ
n_hands = 4  # 何パートの編成に編曲するか

new_score = m21.stream.Score()  # これが最終的に出来上がるやつ
stream = []
for i in range(n_hands):
  stream.append(m21.stream.Part())
  # 楽器
  inst = m21.instrument.Instrument()
  inst.partName = "Piano %s" % i
  stream[i].append(inst)
  # 音部記号
  clef = ''
  if i < 2:
    clef = m21.clef.TrebleClef()
  else:
    clef = m21.clef.BassClef()
  stream[i].append(clef)
  # 調号，拍子
  tempo = 60.0
  key_signature = 0
  time_signature = '4/4'
  for element in s.parts[0].measure(1).elements:  # ()の中は小節数（1スタート）
    if element.__class__.__name__ == "MetronomeMark":
      tempo = element._number
    elif element.__class__.__name__ == "KeySignature":
      key_signature = element._sharps
    elif element.__class__.__name__ == "TimeSignature":
      time_signature = str(element.displaySequence)[1:-1]
  stream[i].append(m21.tempo.MetronomeMark(number=tempo))
  stream[i].append(m21.key.KeySignature(key_signature))
  stream[i].append(m21.meter.TimeSignature(time_signature))
bar_beat_n = int(time_signature[0]) * (4/int(time_signature[-1]))  # 1小節の拍数（4分音符 = 1.0）
global_keySignature = key_signature

bar_interval = 4  # 何小節ごとにやるか
n_clusters = 3    # クラスタ数
n_melody = 6  # 旋律の役割数（ラベルの種類数）
# bar_len = len(s.parts[0])  # ここで小節数が出てくるはずだが...

for barID in range(start_bar, bar_len, bar_interval):
  divided_file = s.measures(barID +1, barID + bar_interval)  # 4小節だけにする
  outputY = getClass(divided_file, n_melody)  # （多分）標準出力したかっただけ？　あとで呼び出しているやつに直アクセスしてるかも
  print(outputY)
  '''
  if barID % 8 == start_bar:  # 2フレーズごとに役割を選択する
    melodies_number = decideMelodies(n_hands, your_difficulty)  # どの旋律を採用するか # ここにEmptyがあるとエラーが出る
  else:  # 2フレーズ目の時
    for m_i in range(len(melodies_number)):
      if not outputY[melodies_number[m_i]]:
        melodies_number[m_i] = 4
  '''
  melodies_number = decideMelodies(n_hands, your_difficulty)  # どの旋律を採用するか # ここにEmptyがあるとエラーが出る
  part_group = getPartGroups(divided_file, melodies_number)

  bar_info = getBarInfo(divided_file, barID, bar_interval)
  all_fhrase_nn, all_length, all_artic, all_tie, all_beat_id, all_bar_id = [], [], [], [], [], []
  for i in range(len(part_group)):
    group = part_group[i]
    
    fhrase_nn, length, artic, tie, beat_id, bar_id = [], [], [], [], [], []
    if group[0] >= 0:
      fhrase_nn = getPhraseNotenum(divided_file, group)
      length, artic, tie, beat_id, bar_id = getPhraseWithoutNotenum(divided_file, group)
    else:
      time_signatures = []
      for info in list(bar_info.values()):
        temp_time_sig = info[2]
        if temp_time_sig and (temp_time_sig != time_signature):
          time_signature = temp_time_sig
        time_signatures.append(time_signature)
      fhrase_nn, length, bar_id = getRest(barID, bar_interval, time_signatures)
    all_fhrase_nn.append(fhrase_nn)  # フレーズのノートナンバーがパート全部入っている
    all_length.append(length)
    all_artic.append(artic)
    all_tie.append(tie)
    all_beat_id.append(beat_id)
    all_bar_id.append(bar_id)

  result_fhrase_nn = adjustPitchRange(all_fhrase_nn, all_beat_id)
  for i in range(len(part_group)):
    inMeas(stream[i], result_fhrase_nn[i], all_length[i], all_artic[i], all_tie[i], all_bar_id[i], bar_info)

for i in range(n_hands):
  new_score.append(stream[i])

result_score = new_score.write('musicxml', "new_%s.xml" % (filename))
print("● 編曲を終わります．お疲れ様でした．")