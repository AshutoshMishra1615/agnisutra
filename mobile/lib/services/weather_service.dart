import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants.dart';

class WeatherData {
  final String day;
  final String date;
  final IconData icon;
  final String description;
  final String temp;

  WeatherData({
    required this.day,
    required this.date,
    required this.icon,
    required this.description,
    required this.temp,
  });
}

class WeatherService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  Future<Map<String, dynamic>?> getBackendWeather(
    double lat,
    double lon,
  ) async {
    try {
      String? token = await _storage.read(key: 'access_token');

      final response = await _dio.get(
        '${AppConstants.baseUrl}/krishi-saathi/weather',
        queryParameters: {"lat": lat, "lon": lon},
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
          },
        ),
      );

      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      print('Backend Weather Service Error: $e');
      return null;
    }
  }

  Future<List<WeatherData>> getWeeklyForecast(double lat, double lon) async {
    try {
      final response = await _dio.get(
        'https://api.open-meteo.com/v1/forecast',
        queryParameters: {
          'latitude': lat,
          'longitude': lon,
          'daily': 'weather_code,temperature_2m_max',
          'timezone': 'auto',
          'forecast_days': 7,
        },
      );

      if (response.statusCode == 200) {
        final data = response.data;
        final daily = data['daily'];
        final List<String> time = List<String>.from(daily['time']);
        final List<int> weatherCodes = List<int>.from(daily['weather_code']);
        final List<double> temps = List<double>.from(
          daily['temperature_2m_max'],
        );

        List<WeatherData> forecast = [];
        for (int i = 0; i < time.length; i++) {
          final date = DateTime.parse(time[i]);
          forecast.add(
            WeatherData(
              day: _getDayName(date.weekday).toUpperCase(),
              date: '${_getMonthName(date.month)} ${date.day}',
              icon: _getIcon(weatherCodes[i]),
              description: _getDescription(weatherCodes[i]),
              temp: temps[i].toStringAsFixed(1),
            ),
          );
        }
        return forecast;
      }
      return [];
    } catch (e, stackTrace) {
      debugPrint('Error fetching weather: $e');
      debugPrint(stackTrace.toString());
      rethrow;
    }
  }

  String _getDayName(int weekday) {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    return days[weekday - 1];
  }

  String _getMonthName(int month) {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    return months[month - 1];
  }

  IconData _getIcon(int code) {
    // WMO Weather interpretation codes (WW)
    if (code == 0) return Icons.wb_sunny;
    if (code >= 1 && code <= 3) return Icons.wb_cloudy;
    if (code >= 45 && code <= 48) return Icons.foggy;
    if (code >= 51 && code <= 67) return Icons.grain; // Drizzle/Rain
    if (code >= 71 && code <= 77) return Icons.ac_unit; // Snow
    if (code >= 80 && code <= 82) return Icons.grain; // Rain showers
    if (code >= 85 && code <= 86) return Icons.ac_unit; // Snow showers
    if (code >= 95 && code <= 99) return Icons.flash_on; // Thunderstorm
    return Icons.wb_sunny;
  }

  String _getDescription(int code) {
    if (code == 0) return 'Clear sky';
    if (code >= 1 && code <= 3) return 'Partly cloudy';
    if (code >= 45 && code <= 48) return 'Fog';
    if (code >= 51 && code <= 67) return 'Rain';
    if (code >= 71 && code <= 77) return 'Snow';
    if (code >= 80 && code <= 82) return 'Rain showers';
    if (code >= 85 && code <= 86) return 'Snow showers';
    if (code >= 95 && code <= 99) return 'Thunderstorm';
    return 'Clear';
  }
}
