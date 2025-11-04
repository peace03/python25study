
# 현재 재고 현황을 화면에 출력
def chk_goods(goods) :

    goods_key = list(goods.keys())
    print("="*30)
    print("현재 재고 현황 \n")
    print("="*30,end="\n")
    
    for i in goods_key :
        
        print(goods[i]['품목']," : ",goods[i]["재고"])
    print("="*30,end="\n")

    
# 현재 재고가 20개 미만인 품목을 검색하여 화면에 출력
# 발주를 하게 되면 재고를 150개로 갱신함
def chk_order(goods) :

    goods_key = list(goods.keys())
    count = 0
    order_list = {}
    print("="*30)
    for i in goods_key :

        if goods[i]['재고'] < 20 :
            print(goods[i]['품목'],'의 재고가',goods[i]['재고'],'개 입니다. 발주주문을 해주세요')
            order_list[i] = goods[i]
            count += 1
    if count == 0 :
        print("발주할 품목이 없습니다\n")
        print("="*30,end="\n")

    else :
        print("="*30)
        select = input("1. 발주 / 2. 취소")
        print("="*30,end="\n")
        
        if select == '1' :
            
            for i in list(order_list.keys()) :
                if i not in list(goods.keys()) :
                    continue
                goods[i]['재고'] = 150
            print("="*30)
        elif select == '2' :
            print("취소 되었습니다")
            
        else :
            print("잘못 입력하셨습니다. 다시 입력하세요")
        print("="*30,end="\n")


def main(goods) :   
                   
    while True :
        print("="*30)
        select = input("1.재고확인 / 2.발주필요품목확인 / 5.종료: ")
        print("="*30,end="\n")
        
        if select == '1' :
            chk_goods(goods)
        elif select == '2' :
            chk_order(goods)
        elif select == '5' :
            break
        else :
            print(" 잘못 입력하셨습니다. 다시 입력 하세요")
            print("="*30,end="\n")