import numpy as np

#############캐나다 산불모형 FMC#############
def compute_fmc(lon, lat, dj, elv=None): 
    """dj = julian day
    """

    if elv is None:
        # 고도 없는 모델
        latn = 46 + 23.4 * np.exp(-0.0360 * (150 - lon)) # 150은 경도 150°W를 기준  
        d0 = 151 * (lat / latn) # 151은 FMC의 최저점(spring dip), 캐나다 모형은 침엽수 잎의 수분함량 기준인데, 우리나라의 값도 확인 필요 
    else:
        # 고도 반영한 모델
        latn = 43 + 33.7 * np.exp(-0.0351 * (150 - lon))
        d0 = 142.1 * (lat / latn) + 0.0172 * elv # 142.1은 FMC의 최저점(spring dip) 

    nd = dj - d0

    if nd < 30: # 잎의 수분이 점차 줄어드는 단계
        fmc = 85 + 0.0189 * nd**2
    elif 30 <= nd < 50: # 최소 수분함량에 도달하며, 가장 건조한 시기 
        fmc = 32.9 + 3.17 * nd - 0.0288 * nd**2
    else: # 새로운 잎이 형성되고 수분이 많아지는 시기로 fmc = 120%로 고정
        fmc = 120

    return fmc

#############산림과학원(2016) FMC#############
def compute_fmc(species_file): 
    fmc_values = np.array([ 
        105.8,  # 0: 강원지방소나무
        110.3,  # 1: 중부지방소나무
        114.8,  # 2: 잣나무
        107.8,  # 3: 리기다소나무
        112.6   # 4: 곰솔 # 다른 수종은? 다른 방법 고민 필요 #
    ])   
    species_file = species_file.astype(int)
    fmc_map = fmc_values[species_file]  # 각 셀에 해당하는 FMC 값 할당
    return fmc_map  # shape = same as species_array


def compute_csi(cbh_file, species_file): #cbh_file, species_file shape = (H, W) 
    CBH = np.load(cbh_file)
    FMC = compute_fmc(species_file)
    csi =  0.001 * np.power(CBH, 1.5) * np.power(460 + 25.9 * FMC, 1.5)
    return csi