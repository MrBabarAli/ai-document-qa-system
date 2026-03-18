# simple_test.py
import sys
import importlib

def check_package(package_name):
    """Check if a package is installed and get its version"""
    try:
        module = importlib.import_module(package_name.replace('-', '_'))
        # Try to get version in different ways
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        else:
            # Try to get version via importlib.metadata (Python 3.8+)
            try:
                from importlib.metadata import version
                version = version(package_name)
            except:
                version = "unknown"
        return True, version
    except ImportError:
        return False, None
    except Exception as e:
        return False, str(e)

print("=" * 60)
print("🔍 PYTHON ENVIRONMENT CHECK")
print("=" * 60)

# Check Python version
print(f"🐍 Python: {sys.version}")
print(f"📂 Path: {sys.executable}")
print()

# Check if virtual environment is active
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f"🔧 Virtual Env: {'✅ Active' if in_venv else '❌ Not active'}")
print()

# Check essential packages
essential_packages = [
    'setuptools',
    'pip',
    'numpy',
    'pandas',
    'PyPDF2',
    'pdfplumber',
    'streamlit',
]

print("📦 Essential Packages:")
print("-" * 40)
for package in essential_packages:
    installed, version = check_package(package)
    status = "✅" if installed else "❌"
    version_str = f"v{version}" if version and installed else ""
    print(f"{status} {package:15} {version_str}")

print("\n" + "=" * 60)

# Optional: Test if we can import your app modules
print("\n📄 Testing App Modules:")
print("-" * 40)

try:
    # Try to import your text_extractor
    import sys
    import os
    sys.path.append(os.getcwd())  # Add current directory to path
    
    from app.text_extractor import TextExtractor
    print("✅ TextExtractor imported successfully")
    
    # Test the extractor
    extractor = TextExtractor()
    test_text = b"Hello, this is a test."
    result = extractor.extract_text(test_text, "txt")
    print(f"✅ Text extraction test: {result[:50]}...")
    
except ImportError as e:
    print(f"❌ Error importing TextExtractor: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

print("=" * 60)