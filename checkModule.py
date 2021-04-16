def check_car_slot(lenght2):
    car_slot = dict()
    with open("carSlot.csv", "r") as f:
        lines = f.read().split("\n")
        for i, line in enumerate(lines):
            x1, x2, slot, type_car = line.split(",")
            car_slot[i + 1] = [float(x1), float(x2), int(slot), type_car]
    car_lenght = dict()
    with open("car_lenght.csv", "r") as f:
        lines = f.read().split("\n")
        for i, line in enumerate(lines):
            real_lenght, lenght, type1, type2 = line.split(",")
            car_lenght[type2] = [float(real_lenght), float(lenght), type1]
    for i in range(0, len(car_slot)-1):
        index = len(car_slot) - i
        x1, x2, slot, type_car = car_slot[index]
        if x1 < float(lenght2) < x2:
            if slot == 1:
                return [car_lenght[type_car][0]], [type_car]
            else:
                type1, type2 = type_car.split("&")
                return [car_lenght[type1][0], car_lenght[type2][0]], [type1, type2]
    return None, None

