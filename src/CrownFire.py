# ===== import libraries =====
# system
import sys
import os
# if error occur, run the line below
# os.environ["GDAL_MEM_ENABLE_OPEN"] = "YES"
# rater & array
import numpy as np
import rasterio
# Spatial data
import geopandas
import shapely
from osgeo import osr, ogr
# visualize
import matplotlib.pyplot as plt
# Load from K-propagator
# kpropa_dir = r"D:/ForestFire/K_Propagator v1.1.4/K_Propagator v1.1.8"
# os.chdir(kpropa_dir)
from K_Propagator.Load_data import coord_conv, get_aoi, load_ftype_shape, load_ftype, load_raster

############################
# 1. 수분함량 계산
############################
class CrownFire:
    def __init__(self):
        self.species_code_name_dict = {10: '기타침엽수', 11: '소나무', 12: '잣나무', 13: '낙엽송',
                                    14: '리기다소나무', 15: '곰솔', 16: '전나무', 17: '편백나무',
                                    18: '삼나무', 19: '가문비나무', 20: '비자나무', 21: '은행나무',
                                    30: '기타활엽수', 31: '상수리나무', 32: '신갈나무', 33: '굴참나무',
                                    34: '기타참나무류', 35: '오리나무', 36: '고로쇠나무', 37: '자작나무',
                                    38: '박달나무', 39: '밤나무', 40: '물푸레나무', 41: '서어나무',
                                    42: '때죽나무', 43: '호두나무', 44: '백합나무', 45: '포플러',
                                    46: '벚나무', 47: '느티나무', 48: '층층나무', 49: '아까시나무',
                                    60: '기타상록활엽수', 61: '가시나무', 62: '구실잣밤나무', 63: '녹나무',
                                    64: '굴거리나무', 65: '황칠나무', 66: '사스레피나무', 67: '후박나무',
                                    68: '새덕이', 77: '침활혼효림'}
        self._proxy_species_dict = None # 대체수종 dictionary 추가 필요
        self.species = None
        self.CBH = None
        self.FMC = None
        self.CSI = None
        self.ROS = None
        self.CFB = None
        self.rso_all = None
        self.rso_masked = None
        self.crowning_mask = None
        self.crowning_type = None

    def _mapping(self, keys, vals, target, default):
        order = np.argsort(keys)
        keys_sorted = keys[order]
        vals_sorted = vals[order]
        if target.ndim > 1:
            flat = target.ravel()
        elif target.ndim == 1:
            flat = target
        else:
            raise ValueError(
                 f"Invalid target dimension: expected 1D or higher, got ndim={target.ndim}"
            )
        p = np.searchsorted(keys_sorted, flat, side="left")
        n = keys_sorted.size
        valid = p < n

        matched = np.zeros_like(flat, dtype=bool)
        matched[valid] = (keys_sorted[p[valid]] == flat[valid])

        out = np.full(flat.shape, default, dtype=vals_sorted.dtype)
        out[matched] = vals_sorted[p[matched]]

        return out.reshape(target.shape)
        

    def compute_fmc(self, species_arr, fmc_default=199.7): 
        """
        species_arr: 수종 코드가 포함된 2D array
        default: 매칭 실패 시 채우는 값 (199.7: 활엽수 평균값 Kozlowski, T.T. and Clausen, J.J. (1965))
        return: 각 셀에 해당하는 FMC 값 (same shape)
        """
        self.species = species_arr        
        fmc_values = np.array([
            105.8,  # 0: 강원지방소나무
            110.3,  # 1: 중부지방소나무
            114.8,  # 2: 잣나무
            107.8,  # 3: 리기다소나무
            112.6   # 4: 곰솔
        ])

        # ===== 대체수종 매핑 =====
        # (수정 필요) map_vals(대체수종)은 임의의 값 혹은 null(-999.)을 넣어둠.
        map_keys = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 30, 31, 32, 33, 34, 
                    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
                    60, 61, 62, 63, 64, 65, 66, 67, 68, 77])
        map_vals = np.array([0, 0, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, -999., -999., -999., -999., -999.
                   -999., -999., -999., -999., -999., -999., -999., -999., -999., -999., -999., -999., -999., -999., -999.,
                   -999., -999., -999., -999., -999., -999., -999., -999., -999., -999., -999.])
        
        species_mapped = self._mapping(map_keys, map_vals, species_arr, -999)
    
        species_mapped = species_mapped.astype(int)
        fmc_map = np.full_like(species_mapped, fill_value=fmc_default, dtype=float)
    
        valid_mask = (species_mapped >= 0) & (species_mapped < len(fmc_values))
        fmc_map[valid_mask] = fmc_values[species_mapped[valid_mask]]
    
        return fmc_map
    
    ############################
    # 2. 수관화 임계강도 계산
    ############################
    
    def compute_csi(self, cbh_arr): #cbh_file, species_file shape = (H, W) 
        """
        cbh_file: NumPy 파일 경로 (CBH 값이 저장됨)
        species_file: 수종 배열 (2D array)
        return: 수관화 임계강도 csi (same shape)
        """
        self.CBH = cbh_arr
        FMC = self.compute_fmc(self.species)
        self.FMC = FMC
        csi = 0.001 * np.power(cbh_arr, 1.5) * np.power(460 + 25.9 * FMC, 1.5) # 0.001 확인 필요(단위 관련)
        self.CSI = csi.astype(float)
        return csi
    
    ############################
    # 3. 수관화 여부 및 ROS 계산
    ############################
    
    def classify_crowning(self, dIntensity):
        """
        dIntensity > csi → 수관화 발생(1), 그렇지 않으면(0)
        """
        dIntensity = dIntensity.astype(float) # 위 코드에서 dIntensity 연결시키기?
        csi = self.CSI
        crowning_mask = (dIntensity > csi).astype(np.uint8) # 수관화 발생 조건: 표면 화재 강도가 임계 강도보다 클 때 # True (1) 수관화 발생/ False (0) 수관화 미발생 #####
        self.crowning_mask = crowning_mask
        return crowning_mask
    
    def compute_rso(self, sfc):
        """
        수관화 발생 셀의 확산속도 계산 (CSI / 300 * SFC)
        """
        sfc = np.asarray(sfc, dtype=float) # 이건 BUI(DMC, DC)와 FFMC로 계산 또는 이미 계산되어 있는 값 활용할지 확인 필요함
        denominator = 300.0 * sfc
        csi = self.CSI
        with np.errstate(divide='ignore', invalid='ignore'):
            rso_all = np.where(denominator > 0, csi / denominator, 0)
        self.rso_all = rso_all
        return rso_all
    
    def compute_masked_rso(self, dIntensity, sfc):
        """
        수관화 발생 셀에 대해서만 rso 계산
        """
        crowning_mask = self.classify_crowning(dIntensity)
        rso_all = self.compute_rso(sfc)
        rso_masked = np.where(crowning_mask == 1, rso_all, 0)
        self.rso_masked, self.crowning_mask = rso_masked, crowning_mask
        return rso_masked, crowning_mask
    
    ############################
    # 4. 수관 연소 분율 및 유형
    ############################
    
    def compute_cfb(self, ros):
        """
        rso_masked: 수관화 셀에 대해 계산된 rso
        ros: 예측된 산불확산 속도(m/min)
        crowning_mask: 수관화 마스크 (0/1)
        return:
            cfb: 수관 연소 분율 (0~1)
            crowning_type: 'active(2)', 'partial(1)', 'none(0)'
        """
        rso, crowning_mask = self.rso_masked, self.crowning_mask
        self.ROS = ros # 위 코드에서 ROS_0 연결시키기?
    
        delta = ros - rso
        cfb = 1.0 - np.exp(-0.23 * delta)
        cfb = np.clip(cfb, 0, 1)
        self.CFB = cfb

        # 0: None, 1: Partial, 2: Active
        crowning_type = np.zeros_like(cfb, dtype=int)
        crowning_type[(crowning_mask == 1) & (cfb >= 0.9)] = 2
        crowning_type[(crowning_mask == 1) & (cfb < 0.9)] = 1
        self.crowning_tpye =crowning_type
    
        return cfb, crowning_type

    ############################
    # 5. Raster Visualization
    ############################ 
    def visualize(arr, title="", unit=""):
        fig, ax = plt.subplots(figsize=(8, 9))
        show(arr, cmap="viridis", ax=ax)
        img = ax.images[0]
        plt.title(f"{title}")
        plt.colorbar(img, ax=ax, label=f"{unit}")
        plt.show()
        
    ############################
    # 6. Run Crownfire Prediction
    ############################ 
    def run (imsang_arr, cbh_arr, dIntensity, sfc, ros):
        # verify the shape
        assert np.shape(imsang_arr) == np.shape(cbh_arr), f"The shape of imsang({np.shape(imsang_arr)}) and CBH({np.shape(cbh_arr)}) requires to be identical."

        self.compute_fmc(imsang_arr) 
        csi = self.compute_csi(cbh_arr)
        self.classify_crowning(dIntensity)
        # 모든 픽셀에 대한 ROS 계산 시,
        rso_all =self.compute_rso(sfc)
        # 수관화 발생(1) 픽셀에서만 ROS 계산 시,
        ros_real = self.compute_masked_rso(dIntensity, sfc)
        cfb, crowning_type = cf.compute_cfb(ros) # rso_real

if __name__ == "__main__":
    # ==== Propgation 내에서 전달받는 변수들 ====
    patch_size = 128
    stride = 18
    resolution = 5.
    long = 128.820
    lat = 36.362
    if (lat < 360) or (lon < 360):
            lat_kor, lon_kor = coord_conv(np.array([lat,long]), 4326, 5179)
    # Boundary 설정을 위한 변수 준비
    # 이전 시점 화선경계(previous_burn)가 없을 경우, boundary 설정 함수 사용
    expand = 0.000001
    burned = shapely.geometry.Point(lon_kor,lat_kor).buffer(zsexpand)
    boundary = get_aoi(burned=burned, patch_size=patch_size, stride=stride, resolution=resolution, center_ignition=[lon_kor, lat_kor])

    # ==== Propgation 지표화 예측 결과 중 사용하는 것들 ====
    dIntensity = np.random.uniform(2000, 80000, size=cbh_arr.size).reshape(np.shape(cbh_arr)) # float, kW/m # 지표화 코드로부터 넘어오는 값
    ros = np.random.uniform(0.5, 20, size=cbh_arr.size).reshape(np.shape(cbh_arr))

    # ==== 수관화 예측을 위한 데이터 준비 ====
    cbh_tif = r"F:\CBH\Clip_HEIGHT_sampled.tif" # r"F:\CBH\result\CBH_meter.tif" #
    imsang = r"F:\ForestFire\Imsang_merge.gdb"
    imsang_arr = load_ftype(boundary, imsang, "KOFTR_GROU") # int
    cbh_arr = load_raster(boundary, cbh_tif) # float, meter
    sfc = np.full_like(cbh_arr, 7.1) # float, kg/m2 # 산과원 자료(영급-경급 코드 매칭)으로 수종별 지표층 연료량 계산 가능

    # ===== 수관화 예측 실행 ====
    cf = CrownFire()
    cf.run(imsang_arr, cbh_arr, dIntensity, sfc, ros)