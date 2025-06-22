import pandas as pd
import os

def is_prime(n):
    """n이 소수인지 판별하는 함수"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def analyze_lotto_history(base_dir):
    """로또 데이터를 읽어와 피처를 추가하고 새로운 CSV 파일로 저장합니다."""
    file_path = os.path.join(base_dir, 'data', 'lotto_history.csv')
    print(f"데이터 파일 경로: {file_path}")

    try:
        print("CSV 파일 읽기를 시도합니다...")
        df = pd.read_csv(file_path)
        print("CSV 파일 읽기 성공!")

        win_cols = ['1', '2', '3', '4', '5', '6']

        print("피처 엔지니어링을 시작합니다...")
        df['sum'] = df[win_cols].sum(axis=1)
        print("- 총합 계산 완료")
        df['odd_count'] = df[win_cols].apply(lambda row: sum(x % 2 != 0 for x in row), axis=1)
        df['even_count'] = 6 - df['odd_count']
        print("- 홀/짝 비율 계산 완료")
        df['high_count'] = df[win_cols].apply(lambda row: sum(x > 22 for x in row), axis=1)
        df['low_count'] = 6 - df['high_count']
        print("- 고/저 비율 계산 완료")

        def count_consecutive(row):
            numbers = sorted(row)
            count = 0
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] == 1:
                    count += 1
            return count
        df['consecutive_count'] = df[win_cols].apply(count_consecutive, axis=1)
        print("- 연속 번호 계산 완료")
        
        df['prime_count'] = df[win_cols].apply(lambda row: sum(is_prime(x) for x in row), axis=1)
        print("- 소수 개수 계산 완료")

        # 1. 끝수 분석 (0~9)
        def get_ending(n):
            return n % 10

        for i in range(10):
            df[f'ending_{i}'] = df[win_cols].apply(lambda row: sum(1 for x in row if get_ending(x) == i), axis=1)
        print("- 끝수 분석 완료")

        # 2. 번호대별 출현 횟수 (1-10, 11-20, 21-30, 31-40, 41-45)
        df['range_1_10'] = df[win_cols].apply(lambda row: sum(1 for x in row if 1 <= x <= 10), axis=1)
        df['range_11_20'] = df[win_cols].apply(lambda row: sum(1 for x in row if 11 <= x <= 20), axis=1)
        df['range_21_30'] = df[win_cols].apply(lambda row: sum(1 for x in row if 21 <= x <= 30), axis=1)
        df['range_31_40'] = df[win_cols].apply(lambda row: sum(1 for x in row if 31 <= x <= 40), axis=1)
        df['range_41_45'] = df[win_cols].apply(lambda row: sum(1 for x in row if 41 <= x <= 45), axis=1)
        print("- 번호대별 출현 횟수 분석 완료")

        output_path = os.path.join(base_dir, 'data', 'lotto_history_featured.csv')
        print(f"새로운 파일을 다음 경로에 저장합니다: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n피처 엔지니어링 완료! 새로운 파일이 '{output_path}'에 저장되었습니다.")
        print("\n새로운 데이터 샘플:")
        print(df.head())

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 스크립트가 프로젝트 루트 디렉토리에서 실행되고 있는지 확인하세요.")
    except Exception as e:
        import traceback
        print(f"오류 발생: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 스크립트가 실행되는 디렉토리를 기준으로 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_lotto_history(current_dir)