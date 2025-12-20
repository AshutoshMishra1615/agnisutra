import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:easy_localization/easy_localization.dart';
import '../services/disease_service.dart';

class HomeCarousel extends StatefulWidget {
  const HomeCarousel({super.key});

  @override
  State<HomeCarousel> createState() => _HomeCarouselState();
}

class _HomeCarouselState extends State<HomeCarousel> {
  final PageController _pageController = PageController();
  final DiseaseService _diseaseService = DiseaseService();
  final ImagePicker _picker = ImagePicker();
  int _currentPage = 0;
  Timer? _timer;
  bool _isUploading = false;

  final List<Map<String, String>> _slides = [
    {'title': 'take_picture', 'image': 'assets/images/home/Focus-cuate 1.png'},
    {
      'title': 'check_diagnose',
      'image': 'assets/images/home/Research paper-cuate 1.png',
    },
    {'title': 'find_solutions', 'image': 'assets/images/home/Key-pana 1.png'},
  ];

  @override
  void initState() {
    super.initState();
    _startAutoScroll();
  }

  void _startAutoScroll() {
    _timer = Timer.periodic(const Duration(seconds: 3), (timer) {
      if (_currentPage < _slides.length - 1) {
        _currentPage++;
      } else {
        _currentPage = 0;
      }

      if (_pageController.hasClients) {
        _pageController.animateToPage(
          _currentPage,
          duration: const Duration(milliseconds: 350),
          curve: Curves.easeIn,
        );
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _pageController.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1E1E1E),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (BuildContext context) {
        return SafeArea(
          child: Wrap(
            children: <Widget>[
              ListTile(
                leading: const Icon(Icons.camera_alt, color: Color(0xFFC5E1A5)),
                title: Text(
                  'take_photo'.tr(),
                  style: const TextStyle(color: Colors.white),
                ),
                onTap: () {
                  Navigator.pop(context);
                  _processImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(
                  Icons.photo_library,
                  color: Color(0xFFC5E1A5),
                ),
                title: Text(
                  'choose_gallery'.tr(),
                  style: const TextStyle(color: Colors.white),
                ),
                onTap: () {
                  Navigator.pop(context);
                  _processImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  Future<Map<String, String>?> _showDetailsDialog() async {
    final cropController = TextEditingController();
    final queryController = TextEditingController();

    return showDialog<Map<String, String>>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: Text(
          'enter_details'.tr(),
          style: const TextStyle(color: Colors.white),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: cropController,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                labelText: 'crop_name'.tr(),
                labelStyle: const TextStyle(color: Colors.white70),
                enabledBorder: const UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.white30),
                ),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: queryController,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                labelText: 'query'.tr(),
                labelStyle: const TextStyle(color: Colors.white70),
                enabledBorder: const UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.white30),
                ),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(
              'cancel'.tr(),
              style: const TextStyle(color: Colors.white70),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, {
              'cropName': cropController.text,
              'query': queryController.text,
            }),
            child: Text(
              'submit'.tr(),
              style: const TextStyle(color: Color(0xFFC5E1A5)),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _processImage(ImageSource source) async {
    try {
      final XFile? photo = await _picker.pickImage(source: source);
      if (photo == null) return;

      final details = await _showDetailsDialog();
      if (details == null) return;

      setState(() {
        _isUploading = true;
      });

      final result = await _diseaseService.predictDisease(
        File(photo.path),
        details['cropName'] ?? '',
        details['query'] ?? '',
      );

      if (mounted) {
        setState(() {
          _isUploading = false;
        });

        if (result != null) {
          _showResultDialog(result);
        } else {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('failed_analyze'.tr())));
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('${'error'.tr()}: $e')));
      }
    }
  }

  void _showResultDialog(Map<String, dynamic> result) {
    String predictedClass = result['predicted_class'] ?? 'Unknown';
    String confidence = result['confidence'] != null 
        ? '${(result['confidence'] * 100).toStringAsFixed(1)}%' 
        : 'N/A';
    String advice = result['preventive_measures'] ?? 'No advice available.';

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: Text(
          'analysis_result'.tr(),
          style: const TextStyle(color: Colors.white),
        ),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '${'Disease'}: $predictedClass', 
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18)
              ),
              const SizedBox(height: 8),
              Text(
                '${'Confidence'}: $confidence', 
                style: const TextStyle(color: Colors.white70)
              ),
              const SizedBox(height: 16),
              Text(
                'Advice:', 
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)
              ),
              const SizedBox(height: 4),
              Text(advice, style: const TextStyle(color: Colors.white)),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(
              'close'.tr(),
              style: const TextStyle(color: Color(0xFFC5E1A5)),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 350, // Adjust height as needed
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
          Expanded(
            child: PageView.builder(
              controller: _pageController,
              onPageChanged: (index) {
                setState(() {
                  _currentPage = index;
                });
              },
              itemCount: _slides.length,
              itemBuilder: (context, index) {
                final slide = _slides[index];
                return Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(height: 24),
                    Text(
                      slide['title']!.tr(),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 24),
                    Expanded(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 24.0),
                        child: Image.asset(
                          slide['image']!,
                          fit: BoxFit.contain,
                          errorBuilder: (context, error, stackTrace) {
                            return const Icon(
                              Icons.image_not_supported,
                              size: 80,
                              color: Colors.white24,
                            );
                          },
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),
                  ],
                );
              },
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(
              _slides.length,
              (index) => AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                margin: const EdgeInsets.symmetric(horizontal: 4),
                height: 8,
                width: 8,
                decoration: BoxDecoration(
                  color: _currentPage == index
                      ? Colors.white
                      : Colors.white.withOpacity(0.4),
                  shape: BoxShape.circle,
                ),
              ),
            ),
          ),
          const SizedBox(height: 24),
          _isUploading
              ? const CircularProgressIndicator(color: Color(0xFFC5E1A5))
              : ElevatedButton.icon(
                  onPressed: _takePicture,
                  icon: const Icon(
                    Icons.camera_alt_outlined,
                    color: Colors.black,
                  ),
                  label: Text(
                    'take_picture'.tr(),
                    style: const TextStyle(
                      color: Colors.black,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFC5E1A5), // Light green
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 32,
                      vertical: 12,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                    elevation: 4,
                  ),
                ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }
}
