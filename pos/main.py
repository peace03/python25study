import pay
import update
import management
import gmanagement
import ep


#재고 및 물품 정보를 텍스트 파일에서 읽기
# goods , day_sale 딕셔너리 생성

f = open("재고/goods.txt","r",encoding="utf-8")
goods ={}                       # 물품 정보 및 재고 저장
day_sale = {"card":0,"cash":0}  # 일 매출 정보 저장

while True:
    tmp_dic = {}
    line = f.readline()
    line = line.rstrip("\n")
    if line== "":
        break
    
    st_list = line.split("/")
    
    tmp_dic["분류"] = st_list[1]
    tmp_dic["품목"] = st_list[2]
    tmp_dic["가격"] = int(st_list[3])
    tmp_dic["재고"] = int(st_list[4])

    goods[st_list[0]] = tmp_dic
    day_sale[st_list[0]] = 0
    

# menu 불러오기

while True:
    print("="*30)
    print("1. 결제 \n2. 물품 관리 \n3. 매출 관리 \n9. 종료")
    print("="*30,end="\n")
    select_num = input('선택 : ')

    # 판매 및 재고, 일매출 정리
    if select_num == '1':
        tmp = pay.main(goods)
        update.main(goods,tmp,day_sale)
        #일매출 텍스트파일이 실행할 때마다 초기화됨

    # 재고 및 발주 관리
    elif select_num == '2':
        gmanagement.main(goods)

    # 일매출 및 월매출 확인
    elif select_num == '3':
        management.main(goods,day_sale)
        #월매출 확인할 때 파일이 없어서 오류남 - 해결!
        #일매출 확인할 때 이전 실행 내용이 안뜸

    # 프로그램 종료 전에 메모리에 있는 정보를 텍스트 파일로 저장
    elif select_num == '9':       
        ep.main(goods,day_sale) #월매출 파일에 계속 업데이트하기 - 해결!
        break
    else:
        print("다시 선택 하세요\n")

print("\nSystem down")
