import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';
import 'package:hive_flutter/hive_flutter.dart';
import '../services/auth_service.dart';
import '../services/weather_service.dart';
// import '../screens/satellite_map_screen.dart';
import '../screens/select_crop_screen.dart';
import '../screens/field_details_screen.dart';
import 'package:latlong2/latlong.dart';
import '../models/crop_data.dart';
import 'home_carousel.dart';

class DashboardWidget extends StatefulWidget {
  const DashboardWidget({super.key});

  @override
  State<DashboardWidget> createState() => _DashboardWidgetState();
}

class _DashboardWidgetState extends State<DashboardWidget> {
  String userName = "User";
  String? _locationName;
  final _authService = AuthService();
  final _weatherService = WeatherService();
  List<WeatherData> _weatherForecast = [];

  final List<Map<String, dynamic>> _cropIcons = const [
    {
      'name': 'Sunflower',
      'icon': 'assets/images/icons/Frame 264.png',
      'color': Color(0xFFEBC25C),
    },
    {
      'name': 'Mustard',
      'icon': 'assets/images/icons/Frame 265.png',
      'color': Color(0xFFEBC25C),
    },
    {
      'name': 'Soyabean',
      'icon': 'assets/images/icons/Frame 266.png',
      'color': Color(0xFF96B65D),
    },
    {
      'name': 'Safflower',
      'icon': 'assets/images/icons/Frame 267.png',
      'color': Color(0xFFEBC25C),
    },
    {
      'name': 'Sesame',
      'icon': 'assets/images/icons/Frame 268.png',
      'color': Color(0xFFC69C6D),
    },
    {
      'name': 'Niger',
      'icon': 'assets/images/icons/Frame 264 (1).png',
      'color': Color(0xFFE0E5C1),
    },
    {
      'name': 'Groundnut',
      'icon': 'assets/images/icons/Frame 267 (1).png',
      'color': Color(0xFFC69C6D),
    },
    {
      'name': 'Castor',
      'icon': 'assets/images/icons/Frame 267 (2).png',
      'color': Color(0xFF96B65D),
    },
  ];

  List<CropData> _crops = [];
  late Box<CropData> _cropsBox;

  @override
  void initState() {
    super.initState();
    _loadUserName();
    _loadWeather();
    _loadCrops();
  }

  Future<void> _loadCrops() async {
    _cropsBox = await Hive.openBox<CropData>('crops');
    if (mounted) {
      setState(() {
        _crops = _cropsBox.values.toList();
      });
    }
  }

  Future<void> _loadUserName() async {
    final name = await _authService.getUserName();
    if (name != null && mounted) {
      setState(() {
        userName = name;
      });
    }
  }

  Future<void> _loadWeather() async {
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        // Fallback to default location (New Delhi) if permission denied
        final forecast = await _weatherService.getWeeklyForecast(
          28.6139,
          77.2090,
        );
        if (mounted) {
          setState(() {
            _weatherForecast = forecast;
            if (forecast.isNotEmpty) {
              final currentTemp = '${forecast.first.temp}°C';
              for (int i = 0; i < _crops.length; i++) {
                final oldCrop = _crops[i];
                _crops[i] = CropData(
                  name: oldCrop.name,
                  statusColor: oldCrop.statusColor,
                  progress: oldCrop.progress,
                  moisture: oldCrop.moisture,
                  temp: currentTemp,
                  sownDate: oldCrop.sownDate,
                  lastIrrigation: oldCrop.lastIrrigation,
                  lastPesticide: oldCrop.lastPesticide,
                  expectedYield: oldCrop.expectedYield,
                );
              }
            }
          });
        }
        return;
      }

      final position = await Geolocator.getCurrentPosition();

      try {
        List<Placemark> placemarks = await placemarkFromCoordinates(
          position.latitude,
          position.longitude,
        );
        if (placemarks.isNotEmpty) {
          final place = placemarks[0];
          if (mounted) {
            setState(() {
              _locationName = "${place.locality}, ${place.administrativeArea}";
            });
          }
        }
      } catch (e) {
        debugPrint("Error getting placemark: $e");
        if (mounted) {
          setState(() {
            _locationName =
                "${position.latitude.toStringAsFixed(2)}, ${position.longitude.toStringAsFixed(2)}";
          });
        }
      }

      final forecast = await _weatherService.getWeeklyForecast(
        position.latitude,
        position.longitude,
      );
      debugPrint('Dashboard Weather Forecast: ${forecast.length} items');
      if (forecast.isNotEmpty) {
        debugPrint('First item temp: ${forecast.first.temp}');
      }
      if (mounted) {
        setState(() {
          _weatherForecast = forecast;
          if (forecast.isNotEmpty) {
            final currentTemp = '${forecast.first.temp}°C';
            for (int i = 0; i < _crops.length; i++) {
              final oldCrop = _crops[i];
              _crops[i] = CropData(
                name: oldCrop.name,
                statusColor: oldCrop.statusColor,
                progress: oldCrop.progress,
                moisture: oldCrop.moisture,
                temp: currentTemp,
                sownDate: oldCrop.sownDate,
                lastIrrigation: oldCrop.lastIrrigation,
                lastPesticide: oldCrop.lastPesticide,
                expectedYield: oldCrop.expectedYield,
                latitude: oldCrop.latitude,
                longitude: oldCrop.longitude,
              );
            }
          }
        });
      }
    } catch (e, stackTrace) {
      debugPrint('Error loading weather for current location: $e');
      debugPrint(stackTrace.toString());

      try {
        // Fallback on error
        final forecast = await _weatherService.getWeeklyForecast(
          28.6139,
          77.2090,
        );
        if (mounted) {
          setState(() {
            _weatherForecast = forecast;
            if (forecast.isNotEmpty) {
              final currentTemp = '${forecast.first.temp}°C';
              for (int i = 0; i < _crops.length; i++) {
                final oldCrop = _crops[i];
                _crops[i] = CropData(
                  name: oldCrop.name,
                  statusColor: oldCrop.statusColor,
                  progress: oldCrop.progress,
                  moisture: oldCrop.moisture,
                  temp: currentTemp,
                  sownDate: oldCrop.sownDate,
                  lastIrrigation: oldCrop.lastIrrigation,
                  lastPesticide: oldCrop.lastPesticide,
                  expectedYield: oldCrop.expectedYield,
                  latitude: oldCrop.latitude,
                  longitude: oldCrop.longitude,
                );
              }
            }
          });
        }
      } catch (e) {
        debugPrint('Error loading fallback weather: $e');
        // If fallback also fails, we can leave _weatherForecast as empty or show error state
      }
    }
  }

  Future<void> _showAddCropDialog() async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const SelectCropScreen()),
    );

    if (result == null) return;

    if (result is CropData) {
      if (!mounted) return;
      await _cropsBox.add(result);
      setState(() {
        _crops.add(result);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF0D0D0D),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              height: 105,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: _crops.length + 1,
                itemBuilder: (context, index) {
                  if (index == _crops.length) {
                    return _buildAddCropCircle();
                  }
                  return _buildCropCircle(_crops[index]);
                },
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'weather_in'.tr(args: [_locationName ?? 'locating'.tr()]),
              style: const TextStyle(color: Colors.white70, fontSize: 14),
            ),
            const SizedBox(height: 8),
            SizedBox(
              height: 110,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: _weatherForecast.length,
                itemBuilder: (context, index) {
                  return _buildWeatherCard(_weatherForecast[index]);
                },
              ),
            ),
            const SizedBox(height: 32),
            const HomeCarousel(),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildAddCropCircle() {
    return GestureDetector(
      onTap: _showAddCropDialog,
      child: Padding(
        padding: const EdgeInsets.only(right: 18.0),
        child: Column(
          children: [
            Container(
              width: 70,
              height: 70,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white54, width: 1),
              ),
              child: const Icon(Icons.add, color: Colors.white54, size: 30),
            ),
            const SizedBox(height: 10),
            Text(
              'add_crop'.tr(),
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCropCircle(CropData crop) {
    final cropInfo = _cropIcons.firstWhere(
      (info) => crop.name.startsWith(info['name'] as String),
      orElse: () => <String, dynamic>{},
    );
    final iconPath = cropInfo['icon'] as String?;

    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => FieldDetailsScreen(
              fieldName: crop.name,
              location: (crop.latitude != null && crop.longitude != null)
                  ? LatLng(crop.latitude!, crop.longitude!)
                  : null,
            ),
          ),
        );
      },
      child: Padding(
        padding: const EdgeInsets.only(right: 16.0),
        child: Column(
          children: [
            Container(
              width: 70,
              height: 70,
              padding: const EdgeInsets.all(0),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                // border: Border.all(color: Colors.amber, width: 2),
              ),
              child: Container(
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  color: Color(0xFF2C2C2C),
                ),
                child: iconPath != null
                    ? ClipOval(
                        child: Image.asset(
                          iconPath,
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) =>
                              const Icon(
                                Icons.local_florist,
                                color: Colors.amber,
                                size: 30,
                              ),
                        ),
                      )
                    : const Icon(
                        Icons.local_florist,
                        color: Colors.amber,
                        size: 30,
                      ),
              ),
            ),
            const SizedBox(height: 10),
            Text(
              crop.name,
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWeatherCard(WeatherData data) {
    return Container(
      width: 80,
      margin: const EdgeInsets.only(right: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(data.icon, color: Colors.amber, size: 28),
          const SizedBox(height: 8),
          Text(
            '${data.temp}°C',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            data.date,
            style: const TextStyle(color: Colors.white70, fontSize: 10),
          ),
          const SizedBox(height: 2),
          Text(
            data.day,
            style: const TextStyle(color: Colors.white70, fontSize: 10),
          ),
        ],
      ),
    );
  }
}
