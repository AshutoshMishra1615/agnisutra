<<<<<<< HEAD
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

      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      // Get address
      try {
        List<Placemark> placemarks = await placemarkFromCoordinates(
          position.latitude,
          position.longitude,
        );
        if (placemarks.isNotEmpty) {
          Placemark place = placemarks[0];
          if (mounted) {
            setState(() {
              _locationName =
                  "${place.subAdministrativeArea ?? place.locality}, ${place.administrativeArea}";
            });
          }
        }
      } catch (e) {
        debugPrint("Error getting address: $e");
      }

      final forecast = await _weatherService.getWeeklyForecast(
        position.latitude,
        position.longitude,
      );

      if (mounted) {
        setState(() {
          _weatherForecast = forecast;
          // Update crop temps
          if (forecast.isNotEmpty) {
            final currentTemp = '${forecast.first.temp}°C';
            // Note: This updates the in-memory list but not Hive.
            // Ideally we should update Hive or just display current temp dynamically.
          }
        });
      }
    } catch (e) {
      debugPrint('Error loading weather: $e');
    }
  }
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
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    }
  }

  Future<void> _showAddCropDialog() async {
<<<<<<< HEAD
    final selectedCropName = await Navigator.push<String>(
=======
    final result = await Navigator.push(
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
      context,
      MaterialPageRoute(builder: (context) => const SelectCropScreen()),
    );

<<<<<<< HEAD
    if (selectedCropName == null) return;

    if (!mounted) return;

    final sownDateController = TextEditingController();
    final lastIrrigationController = TextEditingController();
    final lastPesticideController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF2C3E2D),
        title: Text(
          'Add $selectedCropName Details',
          style: const TextStyle(color: Colors.white),
        ),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildDatePicker(sownDateController, 'Sown Date'),
              _buildDatePicker(lastIrrigationController, 'Last Irrigation'),
              _buildDatePicker(lastPesticideController, 'Last Pesticide'),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text(
              'Cancel',
              style: TextStyle(color: Colors.white70),
            ),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF8F9E8B),
            ),
            onPressed: () {
              setState(() {
                _crops.add(
                  CropData(
                    name: selectedCropName,
                    statusColor: Colors.green,
                    progress: 0,
                    moisture: 'N/A',
                    temp: 'N/A',
                    sownDate: sownDateController.text,
                    lastIrrigation: lastIrrigationController.text,
                    lastPesticide: lastPesticideController.text,
                    expectedYield: 'TBD',
                  ),
                );
              });
              Navigator.pop(context);
            },
            child: const Text('Add', style: TextStyle(color: Colors.black)),
          ),
        ],
      ),
    );
  }

  Widget _buildDatePicker(TextEditingController controller, String label) {
    return TextField(
      controller: controller,
      style: const TextStyle(color: Colors.white),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: Colors.white70),
        enabledBorder: const UnderlineInputBorder(
          borderSide: BorderSide(color: Colors.white54),
        ),
        suffixIcon: const Icon(Icons.calendar_today, color: Colors.white54),
      ),
      readOnly: true,
      onTap: () async {
        DateTime? pickedDate = await showDatePicker(
          context: context,
          initialDate: DateTime.now(),
          firstDate: DateTime(2000),
          lastDate: DateTime(2101),
          builder: (context, child) {
            return Theme(
              data: Theme.of(context).copyWith(
                colorScheme: const ColorScheme.dark(
                  primary: Color(0xFF8F9E8B),
                  onPrimary: Colors.black,
                  surface: Color(0xFF2C3E2D),
                  onSurface: Colors.white,
                ),
              ),
              child: child!,
            );
          },
        );

        if (pickedDate != null) {
          String formattedDate =
              "${pickedDate.day} ${_getMonthName(pickedDate.month)}";
          controller.text = formattedDate;
        }
      },
    );
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
=======
    if (result == null) return;

    if (result is CropData) {
      if (!mounted) return;
      await _cropsBox.add(result);
      setState(() {
        _crops.add(result);
      });
    }
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
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
<<<<<<< HEAD
            const SizedBox(height: 24),
=======
            const SizedBox(height: 16),
            Text(
              'weather_in'.tr(args: [_locationName ?? 'locating'.tr()]),
              style: const TextStyle(color: Colors.white70, fontSize: 14),
            ),
            const SizedBox(height: 8),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
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
<<<<<<< HEAD
            _buildTakePictureCard(),
=======
            const HomeCarousel(),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
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
<<<<<<< HEAD
        padding: const EdgeInsets.only(right: 16.0),
=======
        padding: const EdgeInsets.only(right: 18.0),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
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
<<<<<<< HEAD
            const SizedBox(height: 8),
            const Text(
              'Add Crop',
              style: TextStyle(color: Colors.white, fontSize: 12),
=======
            const SizedBox(height: 10),
            Text(
              'add_crop'.tr(),
              style: const TextStyle(color: Colors.white, fontSize: 16),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCropCircle(CropData crop) {
<<<<<<< HEAD
    return Padding(
      padding: const EdgeInsets.only(right: 16.0),
      child: Column(
        children: [
          Container(
            width: 60,
            height: 60,
            padding: const EdgeInsets.all(2),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: Colors.amber, width: 2),
            ),
            child: Container(
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                color: Color(0xFF2C2C2C),
              ),
              child: const Icon(
                Icons.local_florist,
                color: Colors.amber,
                size: 30,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            crop.name,
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildDot(bool isActive) {
    return Container(
      width: 8,
      height: 8,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: isActive ? const Color(0xFFC5E1A5) : Colors.white24,
=======
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
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
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
<<<<<<< HEAD
            '${data.temp} C',
=======
            '${data.temp}°C',
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
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
<<<<<<< HEAD

  Widget _buildTakePictureCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF2E4E3F), Color(0xFF1B2E24)],
        ),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          const Text(
            'Take a picture',
            style: TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 24),
          SizedBox(
            height: 150,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Icon(
                  Icons.camera_alt_outlined,
                  size: 80,
                  color: Colors.white24,
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildDot(false),
              const SizedBox(width: 8),
              _buildDot(true),
              const SizedBox(width: 8),
              _buildDot(false),
            ],
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () {},
            icon: const Icon(Icons.camera_alt_outlined, color: Colors.black),
            label: const Text(
              'Take a Picture',
              style: TextStyle(color: Colors.black),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFC5E1A5),
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class FlippableCropCard extends StatefulWidget {
  final CropData crop;
  const FlippableCropCard({super.key, required this.crop});

  @override
  State<FlippableCropCard> createState() => _FlippableCropCardState();
}

class _FlippableCropCardState extends State<FlippableCropCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  bool _isFront = true;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOutBack),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _flipCard() {
    if (_isFront) {
      _controller.forward();
    } else {
      _controller.reverse();
    }
    setState(() {
      _isFront = !_isFront;
    });
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        final angle = _animation.value * pi;
        final transform = Matrix4.identity()
          ..setEntry(3, 2, 0.001)
          ..rotateY(angle);

        return Transform(
          transform: transform,
          alignment: Alignment.center,
          child: _animation.value < 0.5
              ? _buildFront()
              : Transform(
                  transform: Matrix4.identity()..rotateY(pi),
                  alignment: Alignment.center,
                  child: _buildBack(),
                ),
        );
      },
    );
  }

  Widget _buildFront() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF4F5E4D), // Muted olive green
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title Bar
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: const Color(0xFF1A261B), // Dark green
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  widget.crop.name,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                Icon(Icons.circle, color: widget.crop.statusColor, size: 12),
              ],
            ),
          ),
          const SizedBox(height: 8),

          // Details Button
          Align(
            alignment: Alignment.centerRight,
            child: GestureDetector(
              onTap: _flipCard,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 4,
                ),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: const Text(
                  'Details',
                  style: TextStyle(
                    color: Colors.black,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),

          const SizedBox(height: 16),
          const Center(
            child: Text(
              'Growth chart',
              style: TextStyle(color: Colors.white, fontSize: 14),
            ),
          ),
          const SizedBox(height: 16),

          // Chart
          SizedBox(
            height: 120,
            child: Stack(
              children: [
                PieChart(
                  PieChartData(
                    sectionsSpace: 0,
                    centerSpaceRadius: 40,
                    startDegreeOffset: 270,
                    sections: [
                      PieChartSectionData(
                        color: const Color(0xFFA8C5A0), // Light sage
                        value: widget.crop.progress,
                        title: '',
                        radius: 20,
                      ),
                      PieChartSectionData(
                        color: const Color(0xFF1A261B), // Dark green
                        value: 100 - widget.crop.progress,
                        title: '',
                        radius: 20,
                      ),
                    ],
                  ),
                ),
                Center(
                  child: Text(
                    '${widget.crop.progress.toInt()}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 20,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Stats
          Text(
            'Soil moisture : ${widget.crop.moisture}',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Text(
            'Needed : 14%',
            style: TextStyle(color: Colors.white70, fontSize: 10),
          ),
          const SizedBox(height: 8),
          Text(
            'Avg Temperature : ${widget.crop.temp}',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Text(
            'Needed : 15-20°C',
            style: TextStyle(color: Colors.white70, fontSize: 10),
          ),
          const SizedBox(height: 24),

          // Dates
          _buildDateRow('Sown Date', widget.crop.sownDate),
          const SizedBox(height: 8),
          _buildDateRow('Last Irrigation', widget.crop.lastIrrigation),
          const SizedBox(height: 8),
          _buildDateRow('Last Pesticide', widget.crop.lastPesticide),
          const SizedBox(height: 8),
          _buildDateRow('Expected Yield', widget.crop.expectedYield),
        ],
      ),
    );
  }

  Widget _buildBack() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF4F5E4D),
        borderRadius: BorderRadius.circular(16),
      ),
      child: SingleChildScrollView(
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '${widget.crop.name} Health',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close, color: Colors.white),
                  onPressed: _flipCard,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(child: _buildHealthBox('Crop Health')),
                const SizedBox(width: 16),
                Expanded(child: _buildHealthBox('Soil Health')),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(child: _buildHealthBox('Irrigation Health')),
                const SizedBox(width: 16),
                Expanded(
                  child: Container(
                    height: 100,
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1A261B),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Text(
                          'Analysis Scale',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 4),
                        const Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Icon(
                              Icons.check_circle,
                              color: Colors.green,
                              size: 16,
                            ),
                            Icon(Icons.error, color: Colors.yellow, size: 16),
                            Icon(Icons.cancel, color: Colors.red, size: 16),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Container(
                          height: 10,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(2),
                            gradient: const LinearGradient(
                              colors: [
                                Colors.green,
                                Colors.yellow,
                                Colors.white,
                                Colors.black,
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildManagementList('Irrigation Management'),
            const SizedBox(height: 16),
            _buildManagementList('Pesticide Management'),
          ],
        ),
      ),
    );
  }

  Widget _buildDateRow(String label, String date) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Flexible(
          child: Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        const SizedBox(width: 8),
        Container(
          constraints: const BoxConstraints(minWidth: 60),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(6),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 2,
                offset: const Offset(0, 1),
              ),
            ],
          ),
          alignment: Alignment.center,
          child: Text(
            date,
            style: const TextStyle(
              color: Colors.black,
              fontSize: 10,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildHealthBox(String title) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: const Color(0xFF1A261B),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(
            title,
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
        ),
        const SizedBox(height: 4),
        Container(
          height: 80,
          decoration: BoxDecoration(
            color: const Color(0xFFD9D9D9),
            borderRadius: BorderRadius.circular(4),
          ),
        ),
      ],
    );
  }

  Widget _buildManagementList(String title) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFF1A261B),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  title,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.white54),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Text(
                'ADD +',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Date',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                'Quantity',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                'Evaporation',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                'Rainfall (mm)',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
=======
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
}
