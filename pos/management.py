#management

import datetime as t


# 월 매출을 파일에서 읽어와서 딕셔너리 형태로 저장한 다음 반환
def month_margin(month) :
    try:
        f = open("관리/"+month+"-total.txt",'r', encoding="utf-8")
    except:
        f = open("관리/"+month+"-total.txt",'w+', encoding="utf-8")
        f.write("""1/0
2/0
3/0
4/0
5/0
6/0
7/0
8/0
9/0
10/0
11/0
card/0
cash/0""")
        f.seek(0)
    tmp_dic = {}
    tmp_month = {}

    while True :
        line = f.readline()
        if line == '' :
            break
        line = line.rstrip('\n')
        tmp_list = line.split('/')
        tmp_month[tmp_list[0]] = int(tmp_list[1])

    tmp_dic[month] = tmp_month
    f.close()
    return tmp_dic


# 받아온 day_sale 딕셔너리를 이용하여 일매출을 화면에 출력
# 월매출 딕셔너리를 반환 받아서 화면에 출력
def main(goods,day_sale) :

    now = t.datetime.now()

    month = now.month
    day = now.day

    if month < 10 :
        month = '0'+str(month)
    else :
        month = str(month)

    if day < 10 :
        day = '0'+str(day)
    else :
        day = str(day)
    

    while True :
        try :
            print("="*30)
            s_num = int(input("1. 일 매출 / 2. 월 매출 / 5. 종료 : "))
            print("="*30,end="\n")

        except :
            continue

        if s_num == 1 :
            print("  일  매  출")
            print("="*30)
            for i in day_sale.keys() :
                if i == 'card' or i == 'cash':
                    continue
                else :
                    
                    print("{}. {} : {}".format(i,goods[i]['품목'],day_sale[i]))
            
            print("="*30,end="\n")
            print("{} : {}\n{} : {}\n".format('card',day_sale['card'],'cash',day_sale['cash']))
            print("="*30,end="\n")
            print()

        elif s_num == 2 :
            
            dic = month_margin(month)
            tmp_dic = dic[month]
            print("월  매  출")
            print("="*30)
            for i in tmp_dic.keys() :
                if i == 'card' or i == 'cash':
                    continue
                else :
                    
                    print("{}. {} : {}".format(i,goods[i]['품목'],tmp_dic[i]))

            print("="*30,end="\n")
            print("{} : {}\n{} : {}\n".format('card',tmp_dic['card'],'cash',tmp_dic['cash']))
            print("="*30,end="\n")
            print()
            
        elif s_num == 5 :
            break

        else :
            print("="*30) 
            print("다시 입력하세요")
            print("="*30,end="\n")
            print()


#main()
