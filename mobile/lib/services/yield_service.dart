import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:hive_flutter/hive_flutter.dart';
import '../constants.dart';

class YieldService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  // 1. Yield Prediction (POST) - Currently maintenance/dummy
  Future<Map<String, dynamic>?> getPrediction(Map<String, dynamic> formData) async {
    var box = await Hive.openBox('last_prediction');
    try {
      String? token = await _storage.read(key: 'access_token');
      
      // Using /krishi-saathi/predict
      final response = await _dio.post(
        '${AppConstants.baseUrl}/krishi-saathi/predict', 
        data: formData,
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
          },
        ),
      );

      if (response.statusCode == 200) {
        await box.put('data', response.data);
        return response.data;
      } else {
        print('Prediction failed: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('Prediction error: $e');
      if (box.isNotEmpty) {
        return Map<String, dynamic>.from(box.get('data'));
      }
      return null;
    }
  }

  // 2. IoT Data
  Future<Map<String, dynamic>?> getIoTData() async {
    try {
      String? token = await _storage.read(key: 'access_token');
      if (token == null) {
        // throw 'User not authenticated. Please log in again.';
        return null;
      }
      
      final response = await _dio.get(
        '${AppConstants.baseUrl}/iot/latest',
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        return response.data;
      } else {
        print('IoT Data failed with status: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('IoT Data error: $e');
      return null;
    }
  }

  // 3. Weather Data
  Future<Map<String, dynamic>?> getWeatherData(double lat, double lon) async {
    try {
      String? token = await _storage.read(key: 'access_token');
      if (token == null) return null;

      final response = await _dio.get(
        '${AppConstants.baseUrl}/krishi-saathi/weather',
        queryParameters: {'lat': lat, 'lon': lon},
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      print('Weather Data error: $e');
      return null;
    }
  }

  // 4. Yield Prediction (GET)
  Future<Map<String, dynamic>?> getYieldPrediction({
    required double nitrogen,
    required double phosphorus,
    required double potassium,
    required double temperature,
    required double humidity,
    required double rainfall,
    required double ph,
    required String crop,
  }) async {
    try {
      String? token = await _storage.read(key: 'access_token');
      
      final response = await _dio.get(
        '${AppConstants.baseUrl}/krishi-saathi/get-yield-prediction',
        queryParameters: {
          'nitrogen': nitrogen,
          'phosphorus': phosphorus,
          'potassium': potassium,
          'temperature': temperature,
          'humidity': humidity,
          'rainfall': rainfall,
          'ph': ph,
          'crop': crop,
        },
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      print('Yield Prediction error: $e');
      return null;
    }
  }

  // 5. NDVI Data
  Future<Map<String, dynamic>?> getNDVI(double lat, double lon) async {
    try {
      final response = await _dio.get(
        '${AppConstants.baseUrl}/krishi-saathi/ndvi',
        queryParameters: {
          'lat': lat,
          'lon': lon,
        },
      );

      if (response.statusCode == 200) {
        return response.data as Map<String, dynamic>;
      }
      return null;
    } catch (e) {
      print('Error fetching NDVI: $e');
      return null;
    }
  }

  // 6. Fertilizer Recommendation
  Future<Map<String, dynamic>?> getFertilizerRecommendation({
    required String crop,
    required double targetYield,
    required double soilN,
    required double soilP,
    required double soilK,
    required double temperature,
    required double moisture,
    double ph = 6.5,
  }) async {
    try {
      String? token = await _storage.read(key: 'access_token');
      
      final response = await _dio.post(
        '${AppConstants.baseUrl}/krishi-saathi/recommend',
        data: {
          "soil_N": soilN,
          "soil_P": soilP,
          "soil_K": soilK,
          "temperature": temperature,
          "humidity": 60.0, // Default if not provided
          "moisture": moisture,
          "ph": ph,
          "crop": crop,
          "target_yield": targetYield
        },
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        var box = await Hive.openBox('last_recommendation');
        await box.put('data', response.data);
        return response.data;
      }
      return null;
    } catch (e) {
      print('Fertilizer Recommendation error: $e');
      return null;
    }
  }
}
