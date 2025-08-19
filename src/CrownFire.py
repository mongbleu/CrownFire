import numpy as np

############################
# 1. 수분함량 계산
############################

def compute_fmc(species_file): 
    """
    species_file: 수종 코드가 포함된 2D array
    return: 각 셀에 해당하는 FMC 값 (same shape)
    """
    fmc_values = np.array([
        105.8,  # 0: 강원지방소나무
        110.3,  # 1: 중부지방소나무
        114.8,  # 2: 잣나무
        107.8,  # 3: 리기다소나무
        112.6   # 4: 곰솔
    ])
    
    default_fmc = 199.7  # 활엽수 평균값 Kozlowski, T.T. and Clausen, J.J. (1965)

    species_file = species_file.astype(int)
    fmc_map = np.full_like(species_file, fill_value=default_fmc, dtype=float)

    valid_mask = (species_file >= 0) & (species_file < len(fmc_values))
    fmc_map[valid_mask] = fmc_values[species_file[valid_mask]]

    return fmc_map

############################
# 2. 수관화 임계강도 계산
############################

def compute_csi(cbh_file, species_file): #cbh_file, species_file shape = (H, W) 
    """
    cbh_file: NumPy 파일 경로 (CBH 값이 저장됨)
    species_file: 수종 배열 (2D array)
    return: 수관화 임계강도 csi (same shape)
    """
    CBH = np.load(cbh_file)
    FMC = compute_fmc(species_file)
    csi = 0.001 * np.power(CBH, 1.5) * np.power(460 + 25.9 * FMC, 1.5) # 0.001 확인 필요(단위 관련)
    return csi

############################
# 3. 수관화 여부 및 ROS 계산
############################

def classify_crowning(dIntensity, csi):
    """
    dIntensity > csi → 수관화 발생(1), 그렇지 않으면(0)
    """
    dIntensity = dIntensity.astype(float) # 위 코드에서 dIntensity 연결시키기?
    csi = csi.astype(float)
    crowning_mask = (dIntensity > csi).astype(np.uint8) # 수관화 발생 조건: 표면 화재 강도가 임계 강도보다 클 때 # True (1) 수관화 발생/ False (0) 수관화 미발생 #####
    return crowning_mask

def compute_ros(csi, sfc):
    """
    수관화 발생 셀의 확산속도 계산 (CSI / 300 * SFC)
    """
    sfc = np.asarray(sfc, dtype=float) # 이건 BUI(DMC, DC)와 FFMC로 계산 또는 이미 계산되어 있는 값 활용할지 확인 필요함
    denominator = 300.0 * sfc

    with np.errstate(divide='ignore', invalid='ignore'):
        ros_all = np.where(denominator > 0, csi / denominator, 0)

    return ros_all

def compute_masked_ros(dIntensity, csi, sfc):
    """
    수관화 발생 셀에 대해서만 ros 계산
    """
    crowning_mask = classify_crowning(dIntensity, csi)
    ros_all = compute_ros(csi, sfc)
    ros_masked = np.where(crowning_mask == 1, ros_all, 0)
    return ros_masked, crowning_mask

############################
# 4. 수관 연소 분율 및 유형
############################

def compute_cfb(ros_masked, rso, crowning_mask):
    """
    ros_masked: 수관화 셀에 대해 계산된 ros
    rso: 수관화 임계속도
    crowning_mask: 수관화 마스크 (0/1)
    return:
        cfb: 수관 연소 분율 (0~1)
        crowning_type: 'active', 'partial', 'none'
    """
    ros = np.asarray(ros_masked, dtype=float)
    rso = np.asarray(rso, dtype=float) # 위 코드에서 ROS_0 연결시키기?

    delta = ros - rso
    cfb = 1.0 - np.exp(-0.23 * delta)
    cfb = np.clip(cfb, 0, 1)

    crowning_type = np.full_like(cfb, fill_value='none', dtype=object)
    crowning_type[(crowning_mask == 1) & (cfb >= 0.9)] = 'active'
    crowning_type[(crowning_mask == 1) & (cfb < 0.9)] = 'partial'

    return cfb, crowning_type
