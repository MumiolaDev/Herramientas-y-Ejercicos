
from typing import List, Tuple, Dict
import json
import math
import numpy as np


def generar_techo(panel_width: int, panel_height: int, ):
    return np.zeros((panel_width,panel_height))

def colocar_panel(techo : np.ndarray, panel_width, panel_height, start_x = 0, start_y = 0, panel_ind=0) :
    tmp = techo.copy()

    for i in range(panel_width):
        for j in range(panel_height):
            if tmp[start_x + i,start_y + j] >= 1:
                raise Exception

            tmp[start_x + i,start_y + j] = 1 + panel_ind
            


    return tmp

def calculate_panels(panel_width: int, panel_height: int, 
                    roof_width: int, roof_height: int) -> int:

    techo = generar_techo(roof_width,roof_height)
    n = 0

    for i in range(roof_width):
        for j in range(roof_height):

            try:
                techo = colocar_panel(techo, panel_width, panel_height,  i, j, n)
                n+=1
            except:
                try:
                    techo = colocar_panel(techo, panel_height, panel_width,  i, j, n)
                    n += 1
                except:
                    pass
            


    return n


def run_tests() -> None:
    with open('test_cases.json', 'r') as f:
        data = json.load(f)
        test_cases: List[Dict[str, int]] = [
            {
                "panel_w": test["panelW"],
                "panel_h": test["panelH"],
                "roof_w": test["roofW"],
                "roof_h": test["roofH"],
                "expected": test["expected"]
            }
            for test in data["testCases"]
        ]
    
    print("Corriendo tests:")
    print("-------------------")
    
    for i, test in enumerate(test_cases, 1):
        result = calculate_panels(
            test["panel_w"], test["panel_h"], 
            test["roof_w"], test["roof_h"]
        )
        passed = result == test["expected"]
        
        print(f"Test {i}:")
        print(f"  Panels: {test['panel_w']}x{test['panel_h']}, "
              f"Roof: {test['roof_w']}x{test['roof_h']}")
        print(f"  Expected: {test['expected']}, Got: {result}")
        print(f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}\n")


def main() -> None:
    print("🐕 Wuuf wuuf wuuf 🐕")
    print("================================\n")
    
    run_tests()


if __name__ == "__main__":
    main()