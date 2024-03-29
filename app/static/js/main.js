const menuBtn = document.querySelector('.menu-btn');
let menuOpen=false;
menuBtn.addEventListener('click', () =>{
    if(!menuOpen){
        menuBtn.classList.add('open');
        menuOpen=true;
    }
    else{
        menuBtn.classList.remove('open');
        menuOpen = false;
    }
});

var $item = $('.carousel .item');
 


var $numberofSlides = $('.item').length;
var $currentSlide = Math.floor((Math.random() * $numberofSlides));

$('.carousel-indicators li').each(function(){
  var $slideValue = $(this).attr('data-slide-to');
  if($currentSlide == $slideValue) {
    $(this).addClass('active');
    $item.eq($slideValue).addClass('active');
  } else {
    $(this).removeClass('active');
    $item.eq($slideValue).removeClass('active');
  }
});

$('.carousel img').each(function() {
  var $src = $(this).attr('src');
  var $color = $(this).attr('data-color');
  $(this).parent().css({
    'background-image' : 'url(' + $src + ')',
    'background-color' : $color,
    'background-repeat':no-repeat
  });
  $(this).remove();
});


$('.carousel').carousel({
  interval: 6000,
  pause: "false"
});